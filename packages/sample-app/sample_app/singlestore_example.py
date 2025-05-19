from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableConfig
from traceloop.sdk.decorators import agent
from traceloop.sdk import Traceloop
from opentelemetry.sdk._logs import LoggingHandler
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry import trace
import logging
import time
import datetime

# Initialize Traceloop
Traceloop.init(app_name="singlestore-agent")

# Create a resource
resource = Resource.create({"service.name": "singlestore-agent"})

# Create a logger provider with the resource
logger_provider = LoggerProvider(resource=resource)

# Create a custom formatter that includes span IDs
class SpanAwareFormatter(logging.Formatter):
    def format(self, record):
        # Get the current span from the context
        current_span = trace.get_current_span()
        
        # Add span ID to the record
        if current_span and current_span.get_span_context().span_id:
            record.span_id = format(current_span.get_span_context().span_id, '016x')
        else:
            record.span_id = 'no-span'

        print("record", record)
            
        return super().format(record)

# Create the handler with the logger provider
handler = LoggingHandler(logger_provider=logger_provider)

# Set up the custom formatter
formatter = SpanAwareFormatter('%(asctime)s - %(name)s - %(levelname)s - [Span: %(span_id)s] - %(message)s')
handler.setFormatter(formatter)

# Use the handler with Python's standard logging
logger = logging.getLogger("myapp")
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# Initialize the LLM with streaming enabled
llm = ChatOpenAI()

# Use the handler with Python's standard logging
logger = logging.getLogger()


@tool("get_funny_current_time", parse_docstring=True)
def get_funny_current_time() -> str:
    """Get the current time with a funny phrase."""
    print("reached here")
    funny_phrase = "according my cat's wristwatch "
    logger.debug("DEBUG LOGS of get_funny_current_time")
    logger.info("INFO LOGS of get_funny_current_time")

    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    funny_timestamp =  f"The time is {current_time}"
    return funny_timestamp


@agent("my_super_agent_new")
def run_agent(question: str):
    agent_executor = create_react_agent(
        tools=[get_funny_current_time],
        model=llm,
        prompt= ChatPromptTemplate.from_messages([
            ("system", "You are a funny AI assistant. Use the tool to get current time and append a funny phrase."),
        ])
    )
    
    return agent_executor.invoke(
        {"input": question},
    )

# Run the agent
if __name__ == "__main__":
    result = run_agent("Please share current time with a funny phrase")
    print(result)
    time.sleep(3)