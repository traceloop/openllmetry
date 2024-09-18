import unittest
import logging
from unittest.mock import Mock
from utils import handle_structured_output  # Ensure this import is correct

# Configure logging to write to a file
logging.basicConfig(
    filename='C:\\openai new feature on onpensource\\openllmetry\\packages\\opentelemetry-instrumentation-openai\\opentelemetry\\instrumentation\\openai\\app.log',  # Name of the log file
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logging.info("Logging configuration set up.")

class TestHandleStructuredOutput(unittest.TestCase):

    def test_handle_refusal_response(self):
        span = Mock()
        response = {
            "choices": [
                {
                    "finish_reason": "refusal",
                    "reason": "safety_violation"
                }
            ]
        }
        handle_structured_output(span, response)
        span.set_attribute.assert_any_call("openai.response_type", "refusal")
        span.set_attribute.assert_any_call("openai.refusal_reason", "safety_violation")
        logging.info("Logged refusal response to span: %s", "safety_violation")

    def test_handle_structured_output(self):
        span = Mock()
        response = {
            "choices": [
                {
                    "structured_output": {
                        "content_filter": "moderate",
                        "action_taken": "flagged"
                    }
                }
            ]
        }
        handle_structured_output(span, response)
        span.set_attribute.assert_any_call("openai.content_filter", "moderate")
        span.set_attribute.assert_any_call("openai.action_taken", "flagged")
        logging.info("Logged structured output: {'content_filter': 'moderate', 'action_taken': 'flagged'}")

    def test_handle_no_refusal_no_structured_output(self):
        span = Mock()
        response = {
            "choices": [
                {
                    "finish_reason": "completed"
                }
            ]
        }
        handle_structured_output(span, response)
        span.set_attribute.assert_not_called()
        logging.info("No refusal or structured output in response.")

if __name__ == '__main__':
    unittest.main()
    print("Logs have been saved to 'app.log'. Check this file for log entries.")

    # Ensure all log entries are flushed and closed
    for handler in logging.root.handlers:
        handler.flush()
        handler.close()
