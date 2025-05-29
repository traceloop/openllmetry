from opentelemetry.semconv_ai import SpanAttributes


def extract_agent_details(test_agent, span):
    if test_agent is None:
        return
    agent = getattr(test_agent, "agent", test_agent)
    if agent is None:
        return

    attributes = {}
    agent_dict = vars(agent)

    for key, value in agent_dict.items():
        if value is not None and isinstance(value, (str, int, float, bool)):
            attributes[f"openai.agent.{key}"] = value
        # Optional: handle known short lists
        elif isinstance(value, list) and len(value) != 0:
            attributes[f"openai.agent.{key}_count"] = len(value)

    if attributes:
        span.set_attributes(attributes)


def set_model_settings_span_attributes(agent, span):

    if not hasattr(agent, "model_settings") or agent.model_settings is None:
        return

    model_settings = agent.model_settings
    settings_dict = vars(model_settings)

    key_to_span_attr = {
        "max_tokens": SpanAttributes.LLM_REQUEST_MAX_TOKENS,
        "temperature": SpanAttributes.LLM_REQUEST_TEMPERATURE,
        "top_p": SpanAttributes.LLM_REQUEST_TOP_P,
    }

    for key, value in settings_dict.items():
        if value is not None:
            span_attr = key_to_span_attr.get(key, f"openai.agent.model.{key}")
            span.set_attribute(span_attr, value)


def extract_run_config_details(run_config, span):
    if run_config is None:
        return

    config_dict = vars(run_config)
    attributes = {}

    for key, value in config_dict.items():

        if value is not None and isinstance(value, (str, int, float, bool)):
            attributes[f"openai.agent.{key}"] = value
        elif isinstance(value, list) and len(value) != 0:
            attributes[f"openai.agent.{key}_count"] = len(value)

    if attributes:
        span.set_attributes(attributes)


def set_prompt_attributes(span, message_history):
    if not message_history:
        return

    for msg in message_history:
        if isinstance(msg, dict):

            role = msg.get("role", "user")
            content = msg.get("content", "")

    span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.role", role)
    span.set_attribute(f"{SpanAttributes.LLM_PROMPTS}.content", content)


def set_response_content_span_attribute(response, span):

    if hasattr(response, "output") and isinstance(response.output, list):
        for output_message in response.output:
            # Extract role and type from output_message
            role = getattr(output_message, "role", None)
            msg_type = getattr(output_message, "type", None)

            if role:
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.role", role)
            if msg_type:
                span.set_attribute(
                    f"{SpanAttributes.LLM_COMPLETIONS}.type", msg_type)

            if hasattr(output_message, "content") and isinstance(
                output_message.content, list
            ):
                for content_item in output_message.content:
                    if hasattr(content_item, "text"):
                        span.set_attribute(
                            f"{SpanAttributes.LLM_COMPLETIONS}.content",
                            content_item.text,
                        )


def set_token_usage_span_attributes(response, span):
    if hasattr(response, "usage"):
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        if input_tokens is not None:
            span.set_attribute(
                SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_tokens)
        if output_tokens is not None:
            span.set_attribute(
                SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, output_tokens
            )
        if total_tokens is not None:
            span.set_attribute(
                SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
