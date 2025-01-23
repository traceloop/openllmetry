import os
from aleph_alpha_client import Client, Prompt, CompletionRequest
from traceloop.sdk import Traceloop

Traceloop.init(disable_batch=True)

def main():
    client = Client(token=os.environ.get("AA_TOKEN"))
    
    # Regular completion
    prompt_text = "Tell me a joke about OpenTelemetry."
    params = {
        "prompt": Prompt.from_text(prompt_text),
        "maximum_tokens": 1000,
    }
    request = CompletionRequest(**params)
    response = client.complete(request, model="luminous-base")
    print(f"Regular completion: {response.completions[0].completion}")

    # Streaming completion
    params["stream"] = True
    request = CompletionRequest(**params)
    response_stream = client.complete(request, model="luminous-base")
    print("\nStreaming completion:")
    for chunk in response_stream:
        if chunk.completions:
            print(chunk.completions[0].completion, end="", flush=True)
    print()

if __name__ == "__main__":
    main()