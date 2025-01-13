import os

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, agent, workflow, tool

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init(app_name="joke_generation_service")


@task(name="joke_creation", version=1)
def create_joke():
    Traceloop.set_prompt(
        "Tell me a joke about {subject}", {"subject": "opentelemetry"}, 1
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    return completion.choices[0].message.content


@agent(name="joke_translation")
def translate_joke_to_pirate(joke: str):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": f"Translate the below joke to pirate-like english:\n\n{joke}",
            }
        ],
    )

    history_jokes_tool()

    return completion.choices[0].message.content


@tool(name="history_jokes")
def history_jokes_tool():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "get some history jokes"}],
    )

    return completion.choices[0].message.content


@task(name="signature_generation")
def generate_signature(joke: str):
    completion = client.completions.create(
        model="davinci-002", prompt="add a signature to the joke:\n\n" + joke
    )

    return completion.choices[0].text


@workflow(name="pirate_joke_generator")
def joke_workflow():
    Traceloop.set_association_properties(
        {"user_id": "user_12345", "chat_id": "chat_1234"}
    )

    eng_joke = create_joke()
    pirate_joke = translate_joke_to_pirate(eng_joke)
    signature = generate_signature(pirate_joke)

    traceloop_client = Traceloop.get()
    traceloop_client.user_feedback.create(
        "sample-annotation-task",
        "user_12345",
        {"sentiment": "positive", "score": 0.95, "tones": ["happy", "surprised"]},
    )

    result = pirate_joke + "\n\n" + signature
    print(result)
    return result


joke_workflow()
