from __future__ import annotations

import threading
import time

from opentelemetry import context as context_api
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode

from opentelemetry.instrumentation.openai.v1.responses_runtime import (
    create_stream_traced_data,
    get_existing_response_data,
)
from opentelemetry.instrumentation.openai.v1.responses_safety import (
    ResponsesStreamingSafety,
    apply_response_completion_safety,
)


class ResponseStreamRuntime:
    def __init__(
        self,
        owner,
        *,
        span,
        start_time,
        request_kwargs,
        tracer,
        traced_data,
        instance,
        traced_data_cls,
        response_store,
        span_name,
        process_input,
        get_tools_from_kwargs,
        prepare_request_attributes,
        set_request_attributes,
        parse_response,
        set_data_attributes,
        cache_legacy_parsed_response,
    ) -> None:
        self._owner = owner
        self._response_store = response_store
        self._span_name = span_name
        self._traced_data_cls = traced_data_cls
        self._process_input = process_input
        self._get_tools_from_kwargs = get_tools_from_kwargs
        self._prepare_request_attributes = prepare_request_attributes
        self._set_request_attributes = set_request_attributes
        self._parse_response = parse_response
        self._set_data_attributes = set_data_attributes
        self._cache_legacy_parsed_response = cache_legacy_parsed_response

        owner._span = span
        owner._start_time = start_time
        owner._request_kwargs = request_kwargs
        owner._tracer = tracer
        owner._instance = instance

        existing_data = get_existing_response_data(response_store, request_kwargs)
        owner._traced_data = traced_data or create_stream_traced_data(
            traced_data_cls,
            existing_data=existing_data,
            request_kwargs=request_kwargs,
            start_time=start_time,
            process_input=process_input,
            get_tools_from_kwargs=get_tools_from_kwargs,
        )

        if owner._span is None and owner._tracer is not None:
            owner._span = owner._tracer.start_span(
                span_name,
                kind=SpanKind.CLIENT,
                start_time=int(owner._traced_data.start_time or start_time or time.time_ns()),
                context=owner._traced_data.trace_context or context_api.get_current(),
            )
            set_request_attributes(
                owner._span,
                prepare_request_attributes(owner._request_kwargs),
                owner._instance,
            )

        owner._complete_response_data = None
        owner._output_text = ""
        owner._streaming_safety = ResponsesStreamingSafety(owner._span)
        owner._cleanup_completed = False
        owner._cleanup_lock = threading.Lock()

    def next(self):
        owner = self._owner
        try:
            chunk = owner.__wrapped__.__next__()
        except StopIteration:
            self.process_complete_response()
            raise
        except Exception as exc:
            self.handle_exception(exc)
            raise
        self.process_chunk(chunk)
        return chunk

    async def anext(self):
        owner = self._owner
        try:
            chunk = await owner.__wrapped__.__anext__()
        except StopAsyncIteration:
            self.process_complete_response()
            raise
        except Exception as exc:
            self.handle_exception(exc)
            raise
        self.process_chunk(chunk)
        return chunk

    def process_chunk(self, chunk):
        owner = self._owner
        chunk = owner._streaming_safety.process_chunk(chunk)
        handled_output_text_chunk = False
        if hasattr(chunk, "type"):
            if chunk.type == "response.output_text.delta":
                if hasattr(chunk, "delta") and chunk.delta:
                    owner._output_text += chunk.delta
                handled_output_text_chunk = True
            elif chunk.type == "response.output_text.done":
                delta = getattr(chunk, "delta", None)
                if delta is not None and hasattr(delta, "text") and delta.text:
                    owner._output_text += delta.text
                handled_output_text_chunk = True
            elif chunk.type == "response.completed" and hasattr(chunk, "response"):
                owner._complete_response_data = chunk.response

        if not handled_output_text_chunk and hasattr(chunk, "delta"):
            if hasattr(chunk.delta, "text") and chunk.delta.text:
                owner._output_text += chunk.delta.text

        if hasattr(chunk, "response") and chunk.response:
            owner._complete_response_data = chunk.response

    def process_complete_response(self):
        owner = self._owner
        with owner._cleanup_lock:
            if owner._cleanup_completed:
                return

            try:
                if owner._complete_response_data:
                    parsed_response = self._parse_response(owner._complete_response_data)

                    owner._traced_data.response_id = parsed_response.id
                    owner._traced_data.response_model = parsed_response.model
                    final_output_text = owner._streaming_safety.aggregated_text()
                    if not final_output_text:
                        final_output_text = owner._streaming_safety.flush_text()
                    owner._traced_data.output_text = final_output_text or owner._output_text

                    if parsed_response.usage:
                        owner._traced_data.usage = parsed_response.usage

                    if parsed_response.output:
                        owner._traced_data.output_blocks = {
                            block.id: block for block in parsed_response.output
                        }
                    masked_output_text = apply_response_completion_safety(
                        owner._span,
                        parsed_response,
                    )
                    if masked_output_text is not None:
                        owner._traced_data.output_text = masked_output_text
                    self._cache_legacy_parsed_response(
                        owner._complete_response_data,
                        parsed_response,
                    )

                    self._response_store[parsed_response.id] = owner._traced_data

                self._set_data_attributes(owner._traced_data, owner._span)
                owner._span.set_status(StatusCode.OK)
                owner._span.end()
                owner._cleanup_completed = True

            except Exception as exc:
                if owner._span and owner._span.is_recording():
                    owner._span.set_attribute(ERROR_TYPE, exc.__class__.__name__)
                    owner._span.set_status(StatusCode.ERROR, str(exc))
                    owner._span.end()
                owner._cleanup_completed = True

    def handle_exception(self, exception):
        owner = self._owner
        with owner._cleanup_lock:
            if owner._cleanup_completed:
                return

            if owner._span and owner._span.is_recording():
                owner._span.set_attribute(ERROR_TYPE, exception.__class__.__name__)
                owner._span.record_exception(exception)
                owner._span.set_status(StatusCode.ERROR, str(exception))
                owner._span.end()

            owner._cleanup_completed = True

    def ensure_cleanup(self):
        owner = self._owner
        with owner._cleanup_lock:
            if owner._cleanup_completed:
                return

            try:
                if owner._span and owner._span.is_recording():
                    self._set_data_attributes(owner._traced_data, owner._span)
                    owner._span.set_status(StatusCode.OK)
                    owner._span.end()

                owner._cleanup_completed = True

            except Exception:
                owner._cleanup_completed = True
