from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import logging

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task
from traceloop.sdk.instruments import Instruments
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

# Initialize Traceloop SDK with console exporter for debugging
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

Traceloop.init(
    app_name="fastapi-openllmetry-example",
    disable_batch=True,
    exporter=ConsoleSpanExporter(),
    instruments={Instruments.FASTAPI}
)

# Create FastAPI app
app = FastAPI(
    title="OpenLLMetry FastAPI Example",
    description="Example FastAPI application with OpenTelemetry tracing and LLM instrumentation",
    version="1.0.0"
)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Initialize OpenAI client
client = OpenAI()


# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 150


class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: Optional[int] = None


class JokeResponse(BaseModel):
    joke: str
    category: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to OpenLLMetry FastAPI Example", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "fastapi-openllmetry-example"}


@task(name="validate_input")
def validate_chat_input(request: ChatRequest):
    """Validate the chat request input."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(request.message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

    return True


@task(name="call_openai_api")
def call_openai_chat(request: ChatRequest):
    """Make a call to OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Provide clear and concise responses."
                },
                {
                    "role": "user",
                    "content": request.message
                }
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        return response
    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate response")


@app.post("/chat", response_model=ChatResponse)
@workflow(name="chat_completion")
async def chat_completion(request: ChatRequest):
    """Generate a chat completion using OpenAI."""
    # Validate input
    validate_chat_input(request)

    # Call OpenAI API
    response = call_openai_chat(request)

    # Extract response content
    content = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else None

    return ChatResponse(
        response=content,
        model=request.model,
        tokens_used=tokens_used
    )


@app.get("/joke/{category}", response_model=JokeResponse)
@workflow(name="joke_generator")
async def generate_joke(category: str):
    """Generate a joke in the specified category."""
    if category not in ["programming", "api", "tech", "general"]:
        raise HTTPException(
            status_code=400,
            detail="Category must be one of: programming, api, tech, general"
        )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Tell me a clean, funny joke about {category}"
                }
            ],
            max_tokens=100,
            temperature=0.9
        )

        joke = response.choices[0].message.content
        return JokeResponse(joke=joke, category=category)

    except Exception as e:
        logger.error(f"Joke generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate joke")


@app.get("/metrics")
async def get_metrics():
    """Get basic metrics about the service."""
    return {
        "service": "fastapi-openllmetry-example",
        "tracing": "enabled",
        "instrumentation": ["fastapi", "openai", "opentelemetry"],
        "endpoints": [
            "/",
            "/health",
            "/chat",
            "/joke/{category}",
            "/metrics"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI application with OpenTelemetry instrumentation")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )