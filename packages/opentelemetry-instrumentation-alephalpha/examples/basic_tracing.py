import os
from aleph_alpha_client import Client, CompletionRequest, Prompt
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor

# 1. Initialize OpenTelemetry Instrumentation
AlephAlphaInstrumentor().instrument()

# 2. Get API Key
api_token = os.getenv("ALEPH_ALPHA_API_KEY")
if not api_token:
    raise ValueError("ALEPH_ALPHA_API_KEY environment variable is missing.")

# 3. Create Aleph Alpha Client
client = Client(token=api_token)

# 4. Send Completion Request
request = CompletionRequest(
    prompt=Prompt.from_text("What is OpenTelemetry?"),
    maximum_tokens=50,
)
response = client.complete(request, model="luminous-base")

# 5. Print Response Output
if response.completions:
    print("\n--- Response ---")
    print(response.completions[0].completion)