import os
import openai

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, agent, workflow, tool

openai.api_key = os.getenv("OPENAI_API_KEY")
Traceloop.init(app_name="joke_generation_service")


@task(name="joke_creation")
def create_joke():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    return completion.choices[0].message.content


@agent(name="joke_translation")
def translate_joke_to_pirate(joke: str):
    completion = openai.ChatCompletion.create(
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
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "get some history jokes"}],
    )

    return completion.choices[0].message.content


@task(name="signature_generation")
def generate_signature(joke: str):
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt="add a signature to the joke:\n\n" + joke,
    )

    return completion.choices[0].text


@workflow(name="pirate_joke_generator")
def joke_workflow():
    Traceloop.set_correlation_id("joke_12345")

    eng_joke = create_joke()
    pirate_joke = translate_joke_to_pirate(eng_joke)
    signature = generate_signature(pirate_joke)
    print(pirate_joke + "\n\n" + signature)


joke_workflow()
