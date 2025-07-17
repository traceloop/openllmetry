import dataclasses
import json
import os
from opentelemetry import context as context_api


def set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _is_truthy(value):
    return str(value).strip().lower() in ("true", "1", "yes", "on")


def should_send_prompts():
    env_setting = os.getenv("TRACELOOP_TRACE_CONTENT", "true")
    override = context_api.get_value("override_enable_content_tracing")
    return _is_truthy(env_setting) or bool(override)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)

        if hasattr(o, "to_json"):
            return o.to_json()

        if hasattr(o, "json"):
            return o.json()

        if hasattr(o, "__class__"):
            return o.__class__.__name__

        return super().default(o)
