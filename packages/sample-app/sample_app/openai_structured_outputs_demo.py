import os
from openai import OpenAI
from pydantic import BaseModel
from traceloop.sdk import Traceloop
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init(
    app_name="structured_outputs_demo",
)


class Joke(BaseModel):
    joke: str
    rating: int


def main():
    print("Making request with structured outputs...")
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
        response_format=Joke,
    )

    print("\n=== Response ===")
    print(f"Joke: {response.choices[0].message.parsed.joke}")
    print(f"Rating: {response.choices[0].message.parsed.rating}")
    print(
        "\n=== Check the span output above for 'gen_ai.request.structured_output_schema' attribute ==="
    )


if __name__ == "__main__":
    main()
