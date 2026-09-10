import os

from groq import Groq
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

# Print traces to the terminal when TRACELOOP_API_KEY is not set.
# Set TRACELOOP_API_KEY to export traces to the Traceloop cloud instead.
init_kwargs = {
    "app_name": "groq_example",
    "disable_batch": True,
}
if not os.getenv("TRACELOOP_API_KEY"):
    init_kwargs["exporter"] = ConsoleSpanExporter()

Traceloop.init(**init_kwargs)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# See https://console.groq.com/docs/models for models available on your account.
MODEL = "llama-3.3-70b-versatile"


@task(name="generate_joke")
def generate_joke():
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model=MODEL,
    )

    return completion.choices[0].message.content


@workflow(name="joke_generator")
def joke_generator():
    joke = generate_joke()
    print(joke)


joke_generator()
