from fastapi import FastAPI
from openai import OpenAI
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

# Initialize Traceloop SDK
Traceloop.init(
    app_name="fastapi-example",
    disable_batch=True,
)

# Create FastAPI app
app = FastAPI()

# Initialize OpenAI client
client = OpenAI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/joke")
@workflow(name="openai_joke_generator")
async def generate_joke():
    """Generate a joke using OpenAI and return it via FastAPI endpoint."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "Tell me a funny joke about APIs and web services"
            }
        ],
        max_tokens=150,
        temperature=0.9
    )

    joke = response.choices[0].message.content
    return {"joke": joke}


@app.get("/story/{topic}")
@workflow(name="openai_story_generator")
async def generate_story(topic: str, length: int = 100):
    """Generate a short story about the given topic."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Write a short story about {topic} in approximately {length} words"
            }
        ],
        max_tokens=length * 2,
        temperature=0.7
    )

    story = response.choices[0].message.content
    return {"topic": topic, "story": story, "requested_length": length}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)