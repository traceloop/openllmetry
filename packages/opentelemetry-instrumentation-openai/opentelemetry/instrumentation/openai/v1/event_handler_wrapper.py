from opentelemetry.instrumentation.openai.shared import (
    _set_span_attribute,
)
from opentelemetry.semconv.ai import SpanAttributes
from openai import AssistantEventHandler
from typing_extensions import override


class EventHandleWrapper(AssistantEventHandler):

    _current_text_index = 0

    def __init__(self, original_handler, span):
        super().__init__()
        self._original_handler = original_handler
        self._span = span

    @override
    def on_end(self):
        self._original_handler.on_end()
        self._span.end()

    @override
    def on_event(self, event):
        self._original_handler.on_event(event)

    @override
    def on_run_step_created(self, run_step):
        self._original_handler.on_run_step_created(run_step)

    @override
    def on_run_step_delta(self, delta, snapshot):
        self._original_handler.on_run_step_delta(delta, snapshot)

    @override
    def on_run_step_done(self, run_step):
        self._original_handler.on_run_step_done(run_step)

    @override
    def on_tool_call_created(self, tool_call):
        self._original_handler.on_tool_call_created(tool_call)

    @override
    def on_tool_call_delta(self, delta, snapshot):
        self._original_handler.on_tool_call_delta(delta, snapshot)

    @override
    def on_tool_call_done(self, tool_call):
        self._original_handler.on_tool_call_done(tool_call)

    @override
    def on_exception(self, exception: Exception):
        self._original_handler.on_exception(exception)

    @override
    def on_timeout(self):
        self._original_handler.on_timeout()

    @override
    def on_message_created(self, message):
        self._original_handler.on_message_created(message)

    @override
    def on_message_delta(self, delta, snapshot):
        self._original_handler.on_message_delta(delta, snapshot)

    @override
    def on_message_done(self, message):
        self._original_handler.on_message_done(message)

    @override
    def on_text_created(self, text):
        self._original_handler.on_text_created(text)

    @override
    def on_text_delta(self, delta, snapshot):
        self._original_handler.on_text_delta(delta, snapshot)

    @override
    def on_text_done(self, text):
        self._original_handler.on_text_done(text)
        _set_span_attribute(
            self._span,
            f"{SpanAttributes.LLM_COMPLETIONS}.{self._current_text_index}.role",
            "assistant",
        )
        _set_span_attribute(
            self._span,
            f"{SpanAttributes.LLM_COMPLETIONS}.{self._current_text_index}.content",
            text.value,
        )

        self._current_text_index += 1

    @override
    def on_image_file_done(self, image_file):
        self._original_handler.on_image_file_done(image_file)
