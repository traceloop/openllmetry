from opentelemetry.instrumentation.ai_providers.utils import create_prompt_event


def wrap_completion(tracer, event_logger, capture_content):
    def wrapper(wrapped, instance, args, kwargs):
        with tracer.start_as_current_span(...) as span:
            # Emit prompt event if not using legacy
            if not instance.use_legacy_attributes:
                event_logger.emit(
                    create_prompt_event(
                        kwargs.get("prompt"),
                        system="anthropic",
                        capture_content=capture_content
                    )
                )
            
            # Existing attribute-based logic if using legacy
            if instance.use_legacy_attributes:
                span.set_attribute("ai.prompt", kwargs.get("prompt"))
                
            # ... rest of the wrapper
    return wrapper 