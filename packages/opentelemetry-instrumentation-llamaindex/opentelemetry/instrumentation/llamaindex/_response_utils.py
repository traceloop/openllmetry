"""Utilities for extracting structured data from LlamaIndex raw responses."""

from dataclasses import dataclass
from typing import Any, List, Optional

from ._message_utils import map_finish_reason

# Map LlamaIndex LLM class names to OTel well-known provider values.
_PROVIDER_MAP = {
    "OpenAI": "openai",
    "AzureOpenAI": "azure.ai.openai",
    "Anthropic": "anthropic",
    "Cohere": "cohere",
    "Groq": "groq",
    "MistralAI": "mistral_ai",
    "Bedrock": "aws.bedrock",
    "Gemini": "gcp.gemini",
    "VertexAI": "gcp.vertex_ai",
    "DeepSeek": "deepseek",
    "Perplexity": "perplexity",
}


@dataclass
class TokenUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


def detect_provider_name(instance_or_class_name: Any) -> Optional[str]:
    """Detect OTel provider name from a LlamaIndex LLM instance or class name string.

    Returns OTel well-known value if available, otherwise lowercase class name.
    Returns None if input is None.
    """
    if instance_or_class_name is None:
        return None
    class_name = (
        instance_or_class_name
        if isinstance(instance_or_class_name, str)
        else instance_or_class_name.__class__.__name__
    )
    return _PROVIDER_MAP.get(class_name, class_name.lower())


def extract_model_from_raw(raw: Any) -> Optional[str]:
    """Extract model name from raw LLM response (object or dict)."""
    if hasattr(raw, "model"):
        return raw.model
    if isinstance(raw, dict):
        return raw.get("model")
    return None


def extract_response_id(raw: Any) -> Optional[str]:
    """Extract response ID from raw LLM response (object or dict)."""
    if hasattr(raw, "id"):
        return raw.id
    if isinstance(raw, dict):
        return raw.get("id")
    return None


def extract_token_usage(raw: Any) -> TokenUsage:
    """Extract token usage from raw response. Handles OpenAI, Cohere, and dict formats."""
    usage = _get_nested(raw, "usage")
    if usage:
        result = _extract_openai_usage(usage)
        if result.input_tokens is not None:
            return result

    meta = _get_nested(raw, "meta")
    if meta:
        return _extract_cohere_usage(meta)

    return TokenUsage()


def _get_nested(obj: Any, key: str) -> Any:
    """Get a nested attribute or dict key from obj."""
    val = getattr(obj, key, None)
    if val is not None:
        return val
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _extract_openai_usage(usage: Any) -> TokenUsage:
    """Extract tokens from OpenAI-style usage object/dict."""
    if hasattr(usage, "completion_tokens"):
        return TokenUsage(
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
    if isinstance(usage, dict):
        return TokenUsage(
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
    return TokenUsage()


def _extract_cohere_usage(meta: Any) -> TokenUsage:
    """Extract tokens from Cohere-style meta.tokens or meta.billed_units."""
    tokens = _get_nested(meta, "tokens")
    if tokens:
        inp = _get_int(tokens, "input_tokens")
        out = _get_int(tokens, "output_tokens")
        if inp is not None:
            return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=_safe_sum(inp, out))

    billed = _get_nested(meta, "billed_units")
    if billed:
        inp = _get_int(billed, "input_tokens")
        out = _get_int(billed, "output_tokens")
        if inp is not None:
            return TokenUsage(input_tokens=inp, output_tokens=out, total_tokens=_safe_sum(inp, out))

    return TokenUsage()


def _get_int(obj: Any, key: str) -> Optional[int]:
    """Get an integer attribute or dict key from obj."""
    val = getattr(obj, key, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(key)
    return int(val) if val is not None else None


def _safe_sum(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is not None and b is not None:
        return a + b
    return None


def extract_finish_reasons(raw: Any) -> List[str]:
    """Extract and map finish reasons from raw LLM response.

    Handles OpenAI choices[], Google Gemini candidates[], Anthropic stop_reason,
    Cohere finish_reason, and Ollama done_reason.
    Returns empty list if no finish reason found.
    """
    if raw is None:
        return []

    # OpenAI format: choices[].finish_reason
    choices = _get_nested(raw, "choices")
    if choices and isinstance(choices, (list, tuple)):
        reasons = _collect_finish_reasons_from_choices(choices)
        if reasons:
            return reasons

    # Google Gemini format: candidates[].finish_reason
    candidates = _get_nested(raw, "candidates")
    if candidates and isinstance(candidates, (list, tuple)):
        reasons = _collect_finish_reasons_from_candidates(candidates)
        if reasons:
            return reasons

    # Anthropic format: stop_reason
    stop_reason = _get_nested(raw, "stop_reason")
    if stop_reason and isinstance(stop_reason, str):
        mapped = map_finish_reason(stop_reason)
        if mapped:
            return [mapped]

    # Cohere / generic: finish_reason (direct attr or in meta)
    fr = _get_nested(raw, "finish_reason")
    if fr and isinstance(fr, str):
        mapped = map_finish_reason(fr)
        if mapped:
            return [mapped]

    # Ollama format: done_reason
    done_reason = _get_nested(raw, "done_reason")
    if done_reason and isinstance(done_reason, str):
        mapped = map_finish_reason(done_reason)
        if mapped:
            return [mapped]

    return []


def _collect_finish_reasons_from_choices(choices: Any) -> List[str]:
    """Collect mapped finish reasons from an OpenAI-style choices array."""
    reasons = []
    try:
        for choice in choices:
            fr = getattr(choice, "finish_reason", None)
            if fr is None and isinstance(choice, dict):
                fr = choice.get("finish_reason")
            mapped = map_finish_reason(fr)
            if mapped:
                reasons.append(mapped)
    except (TypeError, StopIteration):
        pass
    return reasons


def _collect_finish_reasons_from_candidates(candidates: Any) -> List[str]:
    """Collect mapped finish reasons from a Google Gemini-style candidates array."""
    reasons = []
    try:
        for candidate in candidates:
            fr = getattr(candidate, "finish_reason", None)
            if fr is None and isinstance(candidate, dict):
                fr = candidate.get("finish_reason")
            # Gemini finish_reason may be an enum; convert to string name
            if fr is not None and not isinstance(fr, str):
                fr = fr.name if hasattr(fr, "name") else str(fr)
            mapped = map_finish_reason(fr)
            if mapped:
                reasons.append(mapped)
    except (TypeError, StopIteration):
        pass
    return reasons
