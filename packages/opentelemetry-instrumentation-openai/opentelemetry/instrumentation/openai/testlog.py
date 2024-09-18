import logging

# Configure logging with an absolute path
logging.basicConfig(
    filename='C:\\openai new feature on onpensource\\openllmetry\\packages\\opentelemetry-instrumentation-openai\\opentelemetry\\instrumentation\\openai\\app.log',  # Replace with your directory
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logging.info("This is a test log message.")
