"""Event-mode (use_legacy_attributes=False) emission tests."""

from opentelemetry.instrumentation.litellm import _emit_choice_events
from opentelemetry.instrumentation.litellm.config import Config
from opentelemetry.semconv_ai import LLMRequestTypeValues


def test_choice_event_includes_tool_calls(logger_provider, log_exporter, monkeypatch):
    """A gen_ai.choice event must carry the assistant's tool_calls and a mapped
    finish reason. Regression: the event path previously dropped tool_calls
    entirely (ChoiceEvent was built without them), diverging from both the
    attribute path and the OpenAI instrumentor."""
    monkeypatch.setattr(Config, "use_legacy_attributes", False)
    event_logger = logger_provider.get_logger(__name__)

    response_dict = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x": 1}'},
                        }
                    ],
                },
            }
        ]
    }

    _emit_choice_events(LLMRequestTypeValues.CHAT, response_dict, event_logger)

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    body = logs[0].log_record.body
    # tool_calls survive into the event, and the finish reason is mapped.
    assert body["finish_reason"] == "tool_call"
    assert body["tool_calls"][0]["function"]["name"] == "f"
    assert body["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
