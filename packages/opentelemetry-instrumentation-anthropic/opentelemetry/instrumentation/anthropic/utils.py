import asyncio
import json
import logging
import os
import threading
import traceback
from importlib.metadata import version

from opentelemetry import context as context_api
from opentelemetry.instrumentation.anthropic.config import Config
from opentelemetry.semconv_ai import SpanAttributes

GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_SYSTEM_ANTHROPIC = "anthropic"
_PYDANTIC_VERSION = version("pydantic")

TRACELOOP_TRACE_CONTENT = "TRACELOOP_TRACE_CONTENT"


def set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def should_send_prompts():
    return (
        os.getenv(TRACELOOP_TRACE_CONTENT) or "true"
    ).lower() == "true" or context_api.get_value("override_enable_content_tracing")


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.
    Works for both synchronous and asynchronous functions.
    """
    logger = logging.getLogger(func.__module__)

    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            _handle_exception(e, func, logger)

    def _handle_exception(e, func, logger):
        logger.debug(
            "OpenLLMetry failed to trace in %s, error: %s",
            func.__name__,
            traceback.format_exc(),
        )
        if Config.exception_logger:
            Config.exception_logger(e)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


async def _aextract_response_data(response):
    """Async version of _extract_response_data that can await coroutines."""
    import inspect

    # If we get a coroutine, await it
    if inspect.iscoroutine(response):
        try:
            response = await response
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to await coroutine response: {e}")
            return {}

    if isinstance(response, dict):
        return response

    # Handle with_raw_response wrapped responses
    if hasattr(response, 'parse') and callable(response.parse):
        try:
            # For with_raw_response, parse() gives us the actual response object
            parsed_response = response.parse()
            if not isinstance(parsed_response, dict):
                parsed_response = parsed_response.__dict__
            return parsed_response
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to parse response: {e}, response type: {type(response)}")

    # Fallback to __dict__ for regular response objects
    if hasattr(response, '__dict__'):
        response_dict = response.__dict__
        return response_dict

    return {}


def _extract_response_data(response):
    """Extract the actual response data from both regular and with_raw_response wrapped responses."""
    import inspect

    # If we get a coroutine, we cannot process it in sync context
    if inspect.iscoroutine(response):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"_extract_response_data received coroutine {response} - response processing skipped")
        return {}

    if isinstance(response, dict):
        return response

    # Handle with_raw_response wrapped responses
    if hasattr(response, 'parse') and callable(response.parse):
        try:
            # For with_raw_response, parse() gives us the actual response object
            parsed_response = response.parse()
            if not isinstance(parsed_response, dict):
                parsed_response = parsed_response.__dict__
            return parsed_response
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to parse response: {e}, response type: {type(response)}")

    # Fallback to __dict__ for regular response objects
    if hasattr(response, '__dict__'):
        response_dict = response.__dict__
        return response_dict

    return {}


@dont_throw
async def ashared_metrics_attributes(response):
    import inspect

    # If we get a coroutine, await it
    if inspect.iscoroutine(response):
        try:
            response = await response
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to await coroutine response: {e}")
            response = None

    # If it's already a dict (e.g., from streaming), use it directly
    if isinstance(response, dict):
        model = response.get("model")
    else:
        # Handle with_raw_response wrapped responses first
        if response and hasattr(response, "parse") and callable(response.parse):
            try:
                response = response.parse()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to parse with_raw_response: {e}")
                response = None

        # Safely get model attribute without extracting the whole object
        model = getattr(response, "model", None) if response else None

    common_attributes = Config.get_common_metrics_attributes()

    return {
        **common_attributes,
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_ANTHROPIC,
        SpanAttributes.LLM_RESPONSE_MODEL: model,
    }


@dont_throw
def shared_metrics_attributes(response):
    import inspect

    # If we get a coroutine, we cannot process it in sync context
    if inspect.iscoroutine(response):
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"shared_metrics_attributes received coroutine {response} - using None for model")
        response = None

    # If it's already a dict (e.g., from streaming), use it directly
    if isinstance(response, dict):
        model = response.get("model")
    else:
        # Handle with_raw_response wrapped responses first
        if response and hasattr(response, "parse") and callable(response.parse):
            try:
                response = response.parse()
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to parse with_raw_response: {e}")
                response = None

        # Safely get model attribute without extracting the whole object
        model = getattr(response, "model", None) if response else None

    common_attributes = Config.get_common_metrics_attributes()

    return {
        **common_attributes,
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_ANTHROPIC,
        SpanAttributes.LLM_RESPONSE_MODEL: model,
    }


@dont_throw
def error_metrics_attributes(exception):
    return {
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_ANTHROPIC,
        "error.type": exception.__class__.__name__,
    }


@dont_throw
def count_prompt_tokens_from_request(anthropic, request):
    prompt_tokens = 0
    if hasattr(anthropic, "count_tokens"):
        if request.get("prompt"):
            prompt_tokens = anthropic.count_tokens(request.get("prompt"))
        elif messages := request.get("messages"):
            prompt_tokens = 0
            for m in messages:
                content = m.get("content")
                if isinstance(content, str):
                    prompt_tokens += anthropic.count_tokens(content)
                elif isinstance(content, list):
                    for item in content:
                        # TODO: handle image and tool tokens
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt_tokens += anthropic.count_tokens(
                                item.get("text", "")
                            )
    return prompt_tokens


@dont_throw
async def acount_prompt_tokens_from_request(anthropic, request):
    prompt_tokens = 0
    if hasattr(anthropic, "count_tokens"):
        if request.get("prompt"):
            prompt_tokens = await anthropic.count_tokens(request.get("prompt"))
        elif messages := request.get("messages"):
            prompt_tokens = 0
            for m in messages:
                content = m.get("content")
                if isinstance(content, str):
                    prompt_tokens += await anthropic.count_tokens(content)
                elif isinstance(content, list):
                    for item in content:
                        # TODO: handle image and tool tokens
                        if isinstance(item, dict) and item.get("type") == "text":
                            prompt_tokens += await anthropic.count_tokens(
                                item.get("text", "")
                            )
    return prompt_tokens


def run_async(method):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        thread = threading.Thread(target=lambda: asyncio.run(method))
        thread.start()
        thread.join()
    else:
        asyncio.run(method)


def should_emit_events() -> bool:
    """
    Checks if the instrumentation isn't using the legacy attributes
    and if the event logger is not None.
    """
    return not Config.use_legacy_attributes


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_json"):
            return o.to_json()

        if hasattr(o, "model_dump_json"):
            return o.model_dump_json()

        try:
            return str(o)
        except Exception:
            logger = logging.getLogger(__name__)
            logger.debug("Failed to serialize object of type: %s", type(o).__name__)
            return ""


def model_as_dict(model):
    if isinstance(model, dict):
        return model
    if _PYDANTIC_VERSION < "2.0.0" and hasattr(model, "dict"):
        return model.dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    else:
        try:
            return dict(model)
        except Exception:
            return model
