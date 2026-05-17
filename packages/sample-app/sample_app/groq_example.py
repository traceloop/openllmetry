import os

from traceloop.sdk.decorators import task, workflow

from groq import Groq
from traceloop.sdk import Traceloop

Traceloop.init(app_name="groq_example")

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


@task(name="generate_joke")
def generate_joke():
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Tell me a joke about OpenTelemetry",
            }
        ],
        model="llama3-8b-8192",
    )

    return completion.choices[0].message.content


@workflow(name="joke_generator")
def joke_generator():
    joke = generate_joke()
    print(joke)


joke_generator()
