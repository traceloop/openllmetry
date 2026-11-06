import os

import aleph_alpha_client
from aleph_alpha_client import CompletionRequest, Prompt
from opentelemetry.instrumentation.alephalpha import AlephAlphaInstrumentor


# Enable OpenTelemetry instrumentation.
AlephAlphaInstrumentor().instrument()


def main():
    # Create the Aleph Alpha client using the API key.
    client = aleph_alpha_client.Client(
        token=os.environ["ALEPH_ALPHA_API_KEY"]
    )

    # Create a simple prompt.
    prompt = Prompt.from_text(
        "Explain OpenTelemetry in one sentence."
    )

    # Create the completion request.
    request = CompletionRequest(
        prompt=prompt,
        maximum_tokens=100,
    )

    # Make the LLM call.
    response = client.complete(
        request,
        model="luminous-base",
    )

    # Display the response.
    print(response.completions[0].completion)


if __name__ == "__main__":
    main()




