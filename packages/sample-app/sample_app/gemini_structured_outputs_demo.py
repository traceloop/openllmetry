import os
import google.generativeai as genai
from traceloop.sdk import Traceloop
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

Traceloop.init(
    app_name="gemini_structured_outputs_demo",
)


def main():
    print("Making request with structured outputs...")

    response_schema = {
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
        "required": ["joke", "rating"]
    }

    model = genai.GenerativeModel("gemini-1.5-flash")

    result = model.generate_content(
        "Tell me a joke about OpenTelemetry and rate it",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=response_schema
        )
    )

    print("\n=== Response ===")
    print(result.text)
    print("\n=== The 'gen_ai.request.structured_output_schema' attribute should be logged ===")


if __name__ == "__main__":
    main()
