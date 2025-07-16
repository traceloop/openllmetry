import json
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.instrumentation.pinecone.utils import dont_throw, set_span_attribute

@dont_throw
def set_inference_input_attributes(span, method, kwargs):
    #set attributes for inf ops 
    #common attributes
    set_span_attribute(span, SpanAttributes.GEN_AI_SYSTEM, "pinecone")
    set_span_attribute(span, SpanAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))

    if method == "embed":

        #embedding specific attributes
        inputs = kwargs.get("inputs", [])
        set_span_attribute(span, "gen_ai.prompt.count", len(inputs))

        parameters = kwargs.get("parameters", {})
        set_span_attribute(span, "pinecone.inference.input_type", parameters.get("input_type"))
        set_span_attribute(span, "pinecone.inference.truncate", parameters.get("truncate"))

        #token usage estimation
        total_input_tokens = sum(len(str(input_text).split()) for input_text in inputs)
        set_span_attribute(span, SpanAttributes.GEN_AI_INPUT_TOKENS, total_input_tokens)

    elif method == "rerank":
        #reranking specific attributes
        documents = kwargs.get("documents", [])
        set_span_attribute(span, "pinecone.inference.document_count", len(documents))
        set_span_attribute(span, "pinecone.inference.query", kwargs.get("query"))
        set_span_attribute(span, "pinecone.inference.top_k", kwargs.get("top_k"))

@dont_throw
def set_inference_response_attributes(span, method, response):
    #set attributes for inference response
    if method == "embed" and response:

        #embedding response attributes
        data = response.get("data", [])
        set_span_attribute(span, "gen_ai.completion.count", len(data))
        
        usage = response.get("usage", {})
        set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
        
    elif method == "rerank" and response:
        
        #re-ranking response attributes
        data = response.get("data", [])
        set_span_attribute(span, "pinecone.inference.results_count", len(data))
        
        usage = response.get("usage", {})
        set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
    