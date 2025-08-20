from opentelemetry.semconv_ai import SpanAttributes
from enum import Enum
from opentelemetry.instrumentation.bedrock.span_utils import set_guardrail_attributes


class Type(Enum):
    INPUT = "input"
    OUTPUT = "output"


class GuardrailAttributes:
    GUARDRAIL = "gen_ai.guardrail"
    TYPE = "gen_ai.guardrail.type"
    PII = "gen_ai.guardrail.pii"
    PATTERN = "gen_ai.guardrail.pattern"
    TOPIC = "gen_ai.guardrail.topic"
    CONTENT = "gen_ai.guardrail.content"
    CONFIDENCE = "gen_ai.guardrail.confidence"
    MATCH = "gen_ai.guardrail.match"


def is_guardrail_activated(response):
    if "results" in response:
        for message in response["results"]:
            if message.get("completionReason") == "CONTENT_FILTERED":
                return True
    if response.get("stopReason") == "guardrail_intervened":
        return True
    return response.get("amazon-bedrock-guardrailAction") != "NONE"


def handle_invoke_metrics(t: Type, guardrail, attrs, metric_params):
    if "invocationMetrics" in guardrail:
        if "guardrailProcessingLatency" in guardrail["invocationMetrics"]:
            input_latency = guardrail["invocationMetrics"]["guardrailProcessingLatency"]
            metric_params.guardrail_latency_histogram.record(
                input_latency,
                attributes={
                    **attrs,
                    SpanAttributes.LLM_TOKEN_TYPE: t.value,
                },
            )
        if "guardrailCoverage" in guardrail["invocationMetrics"]:
            coverage = guardrail["invocationMetrics"]["guardrailCoverage"]
            char_guarded = coverage["textCharacters"]["guarded"]
            metric_params.guardrail_coverage.add(
                char_guarded,
                attributes={
                    **attrs,
                    SpanAttributes.LLM_TOKEN_TYPE: t.value,
                },
            )


def handle_sensitive(t: Type, guardrail, attrs, metric_params):
    pii = set()
    regex = set()
    if "sensitiveInformationPolicy" in guardrail:
        sensitive_info = guardrail["sensitiveInformationPolicy"]
        if "piiEntities" in sensitive_info:
            for entry in sensitive_info["piiEntities"]:
                pii.add(entry["type"])
                metric_params.guardrail_sensitive_info.add(
                    1,
                    attributes={
                        **attrs,
                        GuardrailAttributes.TYPE: t.value,
                        GuardrailAttributes.PII: entry["type"],
                    },
                )

        if "regexes" in sensitive_info:
            for entry in sensitive_info["regexes"]:
                regex.add(entry["name"])
                metric_params.guardrail_sensitive_info.add(
                    1,
                    attributes={
                        **attrs,
                        GuardrailAttributes.TYPE: t.value,
                        GuardrailAttributes.PATTERN: entry["name"],
                    },
                )
    return {
        "pii": [*pii],
        "regex": [*regex],
    }


def handle_topic(t: Type, guardrail, attrs, metric_params):
    blocked_topics = set()
    if "topicPolicy" in guardrail:
        topics = guardrail["topicPolicy"]["topics"]
        for topic in topics:
            blocked_topics.add(topic["name"])
            metric_params.guardrail_topic.add(
                1,
                attributes={
                    **attrs,
                    GuardrailAttributes.TYPE: t.value,
                    GuardrailAttributes.TOPIC: topic["name"],
                },
            )
    return [*blocked_topics]


def handle_content(t: Type, guardrail, attrs, metric_params):
    content = set()
    if "contentPolicy" in guardrail:
        filters = guardrail["contentPolicy"]["filters"]
        for filter in filters:
            content.add(filter["type"])
            metric_params.guardrail_content.add(
                1,
                attributes={
                    **attrs,
                    GuardrailAttributes.TYPE: t.value,
                    GuardrailAttributes.CONTENT: filter["type"],
                    GuardrailAttributes.CONFIDENCE: filter["confidence"],
                },
            )
    return [*content]


def handle_words(t: Type, guardrail, attrs, metric_params):
    words = set()
    if "wordPolicy" in guardrail:
        filters = guardrail["wordPolicy"]
        if "customWords" in filters:
            for filter in filters["customWords"]:
                words.add(filter["match"])
                metric_params.guardrail_words.add(
                    1,
                    attributes={
                        **attrs,
                        GuardrailAttributes.TYPE: t.value,
                        GuardrailAttributes.MATCH: filter["match"],
                    },
                )
        if "managedWordLists" in filters:
            for filter in filters["managedWordLists"]:
                words.add(filter["match"])
                metric_params.guardrail_words.add(
                    1,
                    attributes={
                        **attrs,
                        GuardrailAttributes.TYPE: t.value,
                        GuardrailAttributes.MATCH: filter["match"],
                    },
                )
    return [*words]


def guardrail_converse(span, response, vendor, model, metric_params):
    attrs = {
        "gen_ai.vendor": vendor,
        SpanAttributes.LLM_RESPONSE_MODEL: model,
        SpanAttributes.LLM_SYSTEM: "bedrock",
    }
    input_filters = None
    output_filters = []
    if "trace" in response and "guardrail" in response["trace"]:
        guardrail = response["trace"]["guardrail"]
        if "inputAssessment" in guardrail:
            guardrail_id = next(iter(guardrail["inputAssessment"]))
            attrs[GuardrailAttributes.GUARDRAIL] = guardrail_id
            guardrail_info = guardrail["inputAssessment"][guardrail_id]
            input_filters = _handle(Type.INPUT, guardrail_info, attrs, metric_params)
        if "outputAssessments" in guardrail:
            guardrail_id = next(iter(guardrail["outputAssessments"]))
            attrs[GuardrailAttributes.GUARDRAIL] = guardrail_id
            guardrail_infos = guardrail["outputAssessments"][guardrail_id]
            for guardrail_info in guardrail_infos:
                output_filters.append(_handle(Type.OUTPUT, guardrail_info, attrs, metric_params))
    if is_guardrail_activated(response):
        metric_params.guardrail_activation.add(1, attrs)
        set_guardrail_attributes(span, input_filters, output_filters)


def guardrail_handling(span, response_body, vendor, model, metric_params):
    input_filters = None
    output_filters = []
    if "amazon-bedrock-guardrailAction" in response_body:
        attrs = {
            "gen_ai.vendor": vendor,
            SpanAttributes.LLM_RESPONSE_MODEL: model,
            SpanAttributes.LLM_SYSTEM: "bedrock",
        }
        if "amazon-bedrock-trace" in response_body:
            bedrock_trace = response_body["amazon-bedrock-trace"]
            if "guardrail" in bedrock_trace and "input" in bedrock_trace["guardrail"]:
                guardrail_id = next(iter(bedrock_trace["guardrail"]["input"]))
                attrs[GuardrailAttributes.GUARDRAIL] = guardrail_id
                guardrail_info = bedrock_trace["guardrail"]["input"][guardrail_id]
                input_filters = _handle(Type.INPUT, guardrail_info, attrs, metric_params)

            if "guardrail" in bedrock_trace and "outputs" in bedrock_trace["guardrail"]:
                outputs = bedrock_trace["guardrail"]["outputs"]
                for output in outputs:
                    guardrail_id = next(iter(output))
                    attrs[GuardrailAttributes.GUARDRAIL] = guardrail_id
                    guardrail_info = outputs[0][guardrail_id]
                    output_filters.append(_handle(Type.OUTPUT, guardrail_info, attrs, metric_params))

        if is_guardrail_activated(response_body):
            metric_params.guardrail_activation.add(1, attrs)
            set_guardrail_attributes(span, input_filters, output_filters)


def _handle(t: Type, guardrail_info, attrs, metric_params):
    handle_invoke_metrics(t, guardrail_info, attrs, metric_params)
    sensitive = handle_sensitive(t, guardrail_info, attrs, metric_params)
    topic = handle_topic(t, guardrail_info, attrs, metric_params)
    content = handle_content(t, guardrail_info, attrs, metric_params)
    words = handle_words(t, guardrail_info, attrs, metric_params)
    return {
        "sensitive": sensitive,
        "topic": topic,
        "content": content,
        "words": words,
    }
