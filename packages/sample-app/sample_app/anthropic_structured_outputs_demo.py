from anthropic import Anthropic
from traceloop.sdk import Traceloop
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

Traceloop.init(
    app_name="anthropic_structured_outputs_demo",
)


def main():
    print("Making request with structured outputs...")

    joke_schema = {
        "type": "object",
        "properties": {
            "joke": {
                "type": "string",
                "description": "A joke about OpenTelemetry"
            },
            "rating": {
                "type": "integer",
                "description": "Rating of the joke from 1 to 10"
            }
        },
        "required": ["joke", "rating"],
        "additionalProperties": False
    }

    response = client.beta.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        betas=["structured-outputs-2025-11-13"],
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry and rate it from 1 to 10"
            }
        ],
        output_format={
            "type": "json_schema",
            "schema": joke_schema
        }
    )

    print("\n=== Response ===")
    print(response.content[0].text)
    print("\n=== The 'gen_ai.request.structured_output_schema' attribute should be logged ===")


if __name__ == "__main__":
    main()
