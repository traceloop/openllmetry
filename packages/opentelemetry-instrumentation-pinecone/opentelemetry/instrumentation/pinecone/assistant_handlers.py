import json
from opentelemetry.semconv_ai import SpanAttributes, Events, EventAttributes
from opentelemetry.instrumentation.pinecone.utils import dont_throw, set_span_attribute

@dont_throw
def set_assistant_input_attributes(span, method, kwargs):
    #set attributes for assistant input
    set_span_attribute(span, SpanAttributes.GEN_AI_SYSTEM, "pinecone")
    
    if method == "create_assistant":
        set_span_attribute(span, "pinecone.assistant.name", kwargs.get("assistant_name"))
        set_span_attribute(span, "pinecone.assistant.instructions", kwargs.get("instructions"))
        set_span_attribute(span, "pinecone.assistant.region", kwargs.get("region"))
        
    elif method == "chat":
        messages = kwargs.get("messages", [])
        set_span_attribute(span, SpanAttributes.GEN_AI_REQUEST_MODEL, kwargs.get("model"))
        set_span_attribute(span, "gen_ai.prompt.count", len(messages))
        
        #log convo messages
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            
            set_span_attribute(span, f"gen_ai.prompt.{i}.role", role)
            set_span_attribute(span, f"gen_ai.prompt.{i}.content", content)
            
        #filter and other parameters
        filter_param = kwargs.get("filter")
        if filter_param:
            set_span_attribute(span, "pinecone.assistant.filter", json.dumps(filter_param))
            
        set_span_attribute(span, "pinecone.assistant.json_response", kwargs.get("json_response", False))
        
    elif method == "upload_file":
        set_span_attribute(span, "pinecone.assistant.file_path", kwargs.get("file_path"))
        metadata = kwargs.get("metadata", {})
        if metadata:
            set_span_attribute(span, "pinecone.assistant.metadata", json.dumps(metadata))

@dont_throw
def set_assistant_response_attributes(span, method, response):
    #set attributes for assistant response
    if method == "chat" and response:
        
        #chat resp attributes
        set_span_attribute(span, SpanAttributes.GEN_AI_RESPONSE_ID, response.get("id"))
        set_span_attribute(span, SpanAttributes.GEN_AI_RESPONSE_MODEL, response.get("model"))
        set_span_attribute(span, "gen_ai.response.finish_reason", response.get("finish_reason"))
        
        #usage information
        usage = response.get("usage", {})
        set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_INPUT_TOKENS, usage.get("prompt_tokens"))
        set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage.get("completion_tokens"))
        set_span_attribute(span, SpanAttributes.GEN_AI_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
        
        #response content
        message = response.get("message", {})
        set_span_attribute(span, "gen_ai.completion.0.role", message.get("role"))
        set_span_attribute(span, "gen_ai.completion.0.content", message.get("content"))
        
        #citations
        citations = response.get("citations", [])
        for i, citation in enumerate(citations):
            set_span_attribute(span, f"pinecone.assistant.citation.{i}.position", citation.get("position"))
            
            references = citation.get("references", [])
            for j, ref in enumerate(references):
                file_info = ref.get("file", {})
                set_span_attribute(span, f"pinecone.assistant.citation.{i}.reference.{j}.file_name", file_info.get("name"))
                set_span_attribute(span, f"pinecone.assistant.citation.{i}.reference.{j}.pages", str(ref.get("pages", [])))
