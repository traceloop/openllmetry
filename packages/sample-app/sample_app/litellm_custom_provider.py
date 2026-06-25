"""LiteLLM custom-provider instrumentation demo.

This is the differentiator for the LiteLLM instrumentation. LiteLLM lets you plug in
your own backend via `litellm.CustomLLM` — for example a provider that POSTs directly
to an internal HTTP endpoint. Such a provider never touches an instrumented SDK (OpenAI,
Anthropic, ...), so SDK-level instrumentation can't see it. Wrapping `litellm.acompletion`
*does* capture it, because every provider — custom ones included — returns a normalized
`ModelResponse`.

This script registers a tiny custom provider that returns a canned response (standing in
for whatever HTTP call a real provider would make) and prints the resulting span to the
console — so it runs with no API key and no network.

Run: uv run python sample_app/litellm_custom_provider.py
"""

import asyncio

import litellm
from litellm import CustomLLM, ModelResponse
from litellm.types.utils import Choices, Message, Usage
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from traceloop.sdk.instruments import Instruments

# Print spans to the console so the demo needs no exporter/backend.
Traceloop.init(
    app_name="litellm_custom_provider_example",
    exporter=ConsoleSpanExporter(),
    instruments={Instruments.LITELLM},
)


class MyCustomLLM(CustomLLM):
    """A minimal custom provider.

    A real provider would build a payload and call an internal HTTP endpoint; here we
    just return a canned, normalized ModelResponse.
    """

    def _response(self, model: str) -> ModelResponse:
        return ModelResponse(
            id="my-custom-resp-1",
            model=model,
            choices=[
                Choices(
                    index=0,
                    finish_reason="stop",
                    message=Message(
                        role="assistant",
                        content="Hello from the custom provider!",
                    ),
                )
            ],
            usage=Usage(prompt_tokens=12, completion_tokens=9, total_tokens=21),
        )

    def completion(self, *args, **kwargs) -> ModelResponse:
        return self._response(kwargs.get("model", "my-custom-llm/demo"))

    async def acompletion(self, *args, **kwargs) -> ModelResponse:
        return self._response(kwargs.get("model", "my-custom-llm/demo"))


# Register the provider under the "my-custom-llm" prefix.
litellm.custom_provider_map = [
    {"provider": "my-custom-llm", "custom_handler": MyCustomLLM()}
]


@task(name="call_custom_provider")
async def call_custom_provider(prompt: str) -> str:
    response = await litellm.acompletion(
        model="my-custom-llm/demo",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


@workflow(name="litellm_custom_provider")
async def main():
    answer = await call_custom_provider("Say hi.")
    print("\nResponse:", answer)
    print(
        "\nLook for a 'litellm.chat' span above with gen_ai.system='my-custom-llm' — "
        "captured even though no provider SDK was involved."
    )


if __name__ == "__main__":
    asyncio.run(main())
