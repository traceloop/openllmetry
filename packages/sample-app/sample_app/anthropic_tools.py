import os
import anthropic
from traceloop.sdk import Traceloop

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

Traceloop.init()

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    tools=[
        {
            "name": "get_property_prices",
            "description": "Get the current average property prices in France",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city, e.g. Paris"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["Euro", "Francs"],
                        "description": "The unit of property prices is either \"Euro\" or \"Francs\""
                    }
                },
                "required": ["location"]
            }
        }
    ],
    messages=[{"role": "user", "content": "What is the current prices in Marseille?"}]
)

print(response)