# test_uninstrument.py
import logging
from opentelemetry.instrumentation.groq import GroqInstrumentor, WRAPPED_METHODS, WRAPPED_AMETHODS
from opentelemetry.instrumentation.utils import unwrap

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_unwrap():
    for wrapped_method in WRAPPED_METHODS + WRAPPED_AMETHODS:
        package = wrapped_method.get("package")
        object_name = wrapped_method.get("object")
        method_name = wrapped_method.get("method")
        try:
            unwrap(
                f"{package}.{object_name}",
                method_name
            )
            logger.info(f"Successfully unwrapped {package}.{object_name}.{method_name}")
        except Exception as e:
            logger.error(f"Failed to unwrap {package}.{object_name}.{method_name}: {e}")
            raise

if __name__ == "__main__":
    test_unwrap()