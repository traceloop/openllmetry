import os

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init(app_name="joke_generation_service")


@task(name="pirate_name_extraction", version=1)
def extract_pirate_name():
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": """
            Instructions: What is the name of the pirate name in the text below?

            Text: Bill the pirate walks into a bar with a steering wheel in his pants. The bartender asks,
            "Hey, what's with the steering wheel in your pants?" The pirate says, "Arrr, it's driving me nuts!"
            """
        }],
    )

    return completion.choices[0].message.content


@task(name="pirate_name_extraction_wrapper", version=1)
def extract_pirate_name_wrapper():
    return extract_pirate_name()


@workflow(name="pirate_name_extraction")
def joke_workflow():
    extract_pirate_name_wrapper()


joke_workflow()
