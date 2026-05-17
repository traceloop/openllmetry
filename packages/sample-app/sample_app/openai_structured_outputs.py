import os
from openai import OpenAI
from pydantic import BaseModel
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init(app_name="story_service")


class StoryStructure(BaseModel):
    setting: str
    protagonist: str
    problem: str
    resolution: str
    moral: str


@task()
def build_joke():
    result = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a story about opentelemetry"}],
    )

    return result.choices[0].message.content


@task()
def build_joke_structure(joke: str):
    result = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "Extract the story structure from the following.",
            },
            {"role": "user", "content": joke},
        ],
        response_format=StoryStructure,
    )

    return result.choices[0].message.parsed


@workflow(name="joke_structure")
def joke_structure():
    joke = build_joke()
    return build_joke_structure(joke)


structure = joke_structure()
print(structure)
