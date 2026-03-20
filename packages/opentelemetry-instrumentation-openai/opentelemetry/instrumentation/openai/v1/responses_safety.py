from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from opentelemetry.instrumentation.fortifyroot import (
    clone_value,
    get_object_value,
    set_object_value,
)
from opentelemetry.instrumentation.fortifyroot.text_streaming import (
    CompletionTextStreamGroup,
)
from opentelemetry.instrumentation.openai.shared.safety_common import (
    CHAT_PROVIDER,
    mask_completion_text,
    mask_prompt_text,
    request_type,
)

SPAN_NAME = "openai.response"


def apply_response_prompt_safety(span, kwargs: dict[str, Any]) -> dict[str, Any]:
    updated_kwargs = kwargs
    changed = False

    instructions = kwargs.get("instructions")
    if isinstance(instructions, str):
        masked_instructions, instructions_changed = mask_prompt_text(
            span,
            instructions,
            span_name=SPAN_NAME,
            segment_index=0,
            segment_role="system",
        )
        if instructions_changed:
            updated_kwargs = dict(updated_kwargs)
            updated_kwargs["instructions"] = masked_instructions
            changed = True

    input_value = kwargs.get("input")
    masked_input, input_changed = _mask_response_input(
        span,
        input_value,
        start_index=1 if isinstance(updated_kwargs.get("instructions"), str) else 0,
    )
    if input_changed:
        if not changed:
            updated_kwargs = dict(updated_kwargs)
        updated_kwargs["input"] = masked_input

    return updated_kwargs


def apply_response_completion_safety(span, response) -> str | None:
    output = get_object_value(response, "output")
    aggregated_text: list[str] = []
    saw_output_text = False
    changed = False

    if isinstance(output, list):
        for output_index, block in enumerate(output):
            if get_object_value(block, "type") != "message":
                continue
            content = get_object_value(block, "content")
            masked_content, block_text, block_changed = _mask_response_output_content(
                span,
                content,
                output_index=output_index,
            )
            if isinstance(block_text, str):
                aggregated_text.append(block_text)
                saw_output_text = True
            if block_changed:
                set_object_value(block, "content", masked_content)
                changed = True

        if saw_output_text:
            aggregated_output_text = "".join(aggregated_text)
            current_output_text = get_object_value(response, "output_text")
            if current_output_text != aggregated_output_text:
                set_object_value(response, "output_text", aggregated_output_text)
                changed = True
            return aggregated_output_text

    output_text = get_object_value(response, "output_text")
    if isinstance(output_text, str):
        masked_output_text, text_changed = mask_completion_text(
            span,
            output_text,
            span_name=SPAN_NAME,
            segment_index=0,
        )
        if text_changed:
            set_object_value(response, "output_text", masked_output_text)
            changed = True
        return masked_output_text if text_changed else output_text

    return None if not changed else ""


class ResponsesStreamingSafety:
    def __init__(self, span):
        self._span = span
        self._streams = CompletionTextStreamGroup(
            span=span,
            provider=CHAT_PROVIDER,
            span_name=SPAN_NAME,
            request_type=request_type(SPAN_NAME),
        )
        self._output_text: dict[tuple[int, int], str] = {}

    def process_chunk(self, chunk):
        chunk_type = get_object_value(chunk, "type")
        if chunk_type == "response.output_text.delta":
            self._process_delta_chunk(chunk)
        elif chunk_type == "response.output_text.done":
            self._process_done_chunk(chunk)
        elif chunk_type == "response.completed":
            self._process_completed_chunk(chunk)
        return chunk

    def flush_text(self) -> str:
        flushed = self._streams.flush_all()
        for key, tail in flushed.items():
            if tail:
                self._output_text[key] = self._output_text.get(key, "") + tail
        return "".join(self._output_text[key] for key in sorted(self._output_text))

    def aggregated_text(self) -> str:
        return "".join(self._output_text[key] for key in sorted(self._output_text))

    def _process_delta_chunk(self, chunk) -> None:
        delta = get_object_value(chunk, "delta")
        if not isinstance(delta, str):
            delta_object = get_object_value(chunk, "delta")
            delta = get_object_value(delta_object, "text")
            if not isinstance(delta, str):
                return
            masked = self._mask_text_chunk(chunk, delta)
            set_object_value(delta_object, "text", masked)
            return

        masked = self._mask_text_chunk(chunk, delta)
        set_object_value(chunk, "delta", masked)

    def _process_done_chunk(self, chunk) -> None:
        key = _response_output_key(chunk)
        tail = self._streams.flush(key)
        if tail:
            self._output_text[key] = self._output_text.get(key, "") + tail
            set_object_value(chunk, "delta", SimpleNamespace(text=tail))

        text = get_object_value(chunk, "text")
        if isinstance(text, str):
            masked_text = self._output_text.get(key, text)
            set_object_value(chunk, "text", masked_text)

    def _process_completed_chunk(self, chunk) -> None:
        self.flush_text()
        response = get_object_value(chunk, "response")
        if response is None:
            return
        masked_output_text = apply_response_completion_safety(self._span, response)
        if masked_output_text is not None:
            self._sync_aggregated_text_from_response(response, masked_output_text)

    def _mask_text_chunk(self, chunk, text: str) -> str:
        output_index = get_object_value(chunk, "output_index", 0) or 0
        content_index = get_object_value(chunk, "content_index", 0) or 0
        key = (output_index, content_index)
        masked = self._streams.process(
            key,
            text,
            segment_index=output_index,
            segment_role="assistant",
            metadata={"content_index": content_index},
        )
        self._output_text[key] = self._output_text.get(key, "") + masked
        return masked

    def _sync_aggregated_text_from_response(self, response, masked_output_text: str) -> None:
        if masked_output_text:
            self._output_text[(0, 0)] = masked_output_text
            return

        output = get_object_value(response, "output")
        if not isinstance(output, list):
            return
        rebuilt: dict[tuple[int, int], str] = {}
        for output_index, block in enumerate(output):
            if get_object_value(block, "type") != "message":
                continue
            content = get_object_value(block, "content")
            if not isinstance(content, list):
                continue
            for content_index, item in enumerate(content):
                text = get_object_value(item, "text")
                if isinstance(text, str):
                    rebuilt[(output_index, content_index)] = text
        if rebuilt:
            self._output_text = rebuilt


def _mask_response_input(span, input_value, *, start_index: int):
    if isinstance(input_value, str):
        return mask_prompt_text(
            span,
            input_value,
            span_name=SPAN_NAME,
            segment_index=start_index,
            segment_role="user",
        )

    if not isinstance(input_value, list):
        return input_value, False

    updated_input = input_value
    changed = False

    for item_index, item in enumerate(input_value):
        masked_item, item_changed = _mask_response_input_item(
            span,
            item,
            segment_index=start_index + item_index,
        )
        if not item_changed:
            continue
        if updated_input is input_value:
            updated_input = clone_value(input_value)
        updated_input[item_index] = masked_item
        changed = True

    return updated_input, changed


def _mask_response_input_item(span, item, *, segment_index: int):
    updated_item = item
    changed = False
    role = str(get_object_value(item, "role", "user") or "user").lower()

    content = get_object_value(item, "content")
    masked_content, content_changed = _mask_prompt_content_blocks(
        span,
        content,
        segment_index=segment_index,
        segment_role=role,
    )
    if content_changed:
        if updated_item is item:
            updated_item = clone_value(item)
        set_object_value(updated_item, "content", masked_content)
        changed = True

    text = get_object_value(item, "text")
    if isinstance(text, str):
        masked_text, text_changed = mask_prompt_text(
            span,
            text,
            span_name=SPAN_NAME,
            segment_index=segment_index,
            segment_role=role,
        )
        if text_changed:
            if updated_item is item:
                updated_item = clone_value(item)
            set_object_value(updated_item, "text", masked_text)
            changed = True

    output_value = get_object_value(item, "output")
    if isinstance(output_value, str):
        masked_output, output_changed = mask_prompt_text(
            span,
            output_value,
            span_name=SPAN_NAME,
            segment_index=segment_index,
            segment_role=role,
        )
        if output_changed:
            if updated_item is item:
                updated_item = clone_value(item)
            set_object_value(updated_item, "output", masked_output)
            changed = True

    return updated_item, changed


def _mask_prompt_content_blocks(span, content, *, segment_index: int, segment_role: str):
    if isinstance(content, str):
        return mask_prompt_text(
            span,
            content,
            span_name=SPAN_NAME,
            segment_index=segment_index,
            segment_role=segment_role,
        )

    if not isinstance(content, list):
        return content, False

    updated_content = content
    changed = False

    for block_index, block in enumerate(content):
        if isinstance(block, str):
            masked_text, text_changed = mask_prompt_text(
                span,
                block,
                span_name=SPAN_NAME,
                segment_index=segment_index,
                segment_role=segment_role,
                metadata={"block_index": block_index},
            )
            if not text_changed:
                continue
            if updated_content is content:
                updated_content = clone_value(content)
            updated_content[block_index] = masked_text
            changed = True
            continue

        text = get_object_value(block, "text")
        if not isinstance(text, str):
            continue
        masked_text, text_changed = mask_prompt_text(
            span,
            text,
            span_name=SPAN_NAME,
            segment_index=segment_index,
            segment_role=segment_role,
            metadata={"block_index": block_index},
        )
        if not text_changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        set_object_value(updated_content[block_index], "text", masked_text)
        changed = True

    return updated_content, changed


def _mask_response_output_content(span, content, *, output_index: int):
    if not isinstance(content, list):
        return content, "", False

    updated_content = content
    changed = False
    aggregated_text: list[str] = []

    for content_index, item in enumerate(content):
        text = get_object_value(item, "text")
        if not isinstance(text, str):
            continue
        masked_text, text_changed = mask_completion_text(
            span,
            text,
            span_name=SPAN_NAME,
            segment_index=output_index,
            metadata={"content_index": content_index},
        )
        aggregated_text.append(masked_text if text_changed else text)
        if not text_changed:
            continue
        if updated_content is content:
            updated_content = clone_value(content)
        set_object_value(updated_content[content_index], "text", masked_text)
        changed = True

    return updated_content, "".join(aggregated_text), changed


def _response_output_key(chunk) -> tuple[int, int]:
    return (
        get_object_value(chunk, "output_index", 0) or 0,
        get_object_value(chunk, "content_index", 0) or 0,
    )
