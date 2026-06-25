"""Basic LiteLLM completion instrumented by Traceloop.

Demonstrates the `opentelemetry-instrumentation-litellm` package: enabling
`Instruments.LITELLM` wraps `litellm.completion` in-process, emitting a single
`litellm.chat` span with `gen_ai.*` attributes regardless of which provider LiteLLM
routes to.

Requires: OPENAI_API_KEY
"""

import litellm
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from traceloop.sdk.instruments import Instruments

Traceloop.init(
    app_name="litellm_completion_example",
    instruments={Instruments.LITELLM},
)


@task(name="answer_question")
def answer_question(question: str) -> str:
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
        temperature=0.7,
    )
    return response.choices[0].message.content


@workflow(name="litellm_completion")
def main():
    answer = answer_question("What is OpenTelemetry in one sentence?")
    print(answer)


if __name__ == "__main__":
    main()
