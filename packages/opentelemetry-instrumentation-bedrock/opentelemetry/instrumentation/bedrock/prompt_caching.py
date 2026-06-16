from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


class CachingHeaders:
    READ = "x-amzn-bedrock-cache-read-input-token-count"
    WRITE = "x-amzn-bedrock-cache-write-input-token-count"


class CacheSpanAttrs:
    TYPE = "gen_ai.cache.type"
    # CACHED is a lossy "read"/"write" marker (the two branches overwrite each other
    # when a single call both reads and writes cache). Superseded by
    # SpanAttributes.GEN_AI_USAGE_CACHE_{READ,CREATION}_INPUT_TOKENS, which carries
    # full numeric counts without collision. Slated for removal in a future major version.
    CACHED = "gen_ai.prompt_caching"


def prompt_caching_handling(headers, vendor, model, metric_params):
    base_attrs = {
        GenAIAttributes.GEN_AI_PROVIDER_NAME: vendor,
        GenAIAttributes.GEN_AI_RESPONSE_MODEL: model,
    }
    span = trace.get_current_span()
    if not isinstance(span, trace.Span):
        return
    read_cached_tokens = None
    write_cached_tokens = None
    if CachingHeaders.READ in headers:
        read_cached_tokens = int(headers[CachingHeaders.READ])
        metric_params.prompt_caching.add(
            read_cached_tokens,
            attributes={
                **base_attrs,
                CacheSpanAttrs.TYPE: "read",
            },
        )
        span.set_attribute(
            GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, read_cached_tokens
        )
    if CachingHeaders.WRITE in headers:
        write_cached_tokens = int(headers[CachingHeaders.WRITE])
        metric_params.prompt_caching.add(
            write_cached_tokens,
            attributes={
                **base_attrs,
                CacheSpanAttrs.TYPE: "write",
            },
        )
        span.set_attribute(
            GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
            write_cached_tokens,
        )
    if write_cached_tokens is not None or read_cached_tokens is not None:
        if (write_cached_tokens or 0) >= (read_cached_tokens or 0):
            span.set_attribute(CacheSpanAttrs.CACHED, "write")
        else:
            span.set_attribute(CacheSpanAttrs.CACHED, "read")
