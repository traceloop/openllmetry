import os
from openai import AzureOpenAI
from traceloop.sdk import Traceloop

Traceloop.init()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_ID")

response = client.chat.completions.create(
    model=deployment_name,
    messages=[{"role": "user", "content": "Tell me a joke about OpenTelemetry"}],
    max_tokens=50,
)
print(response.choices[0].message.content)
