import os

from openai import OpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import agent, workflow

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Traceloop.init(app_name="joke_generation_service")


@agent(name="base_joke_generator", method_name="generate_joke")
class JokeAgent:
    def generate_joke(self):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about Donald Trump"}],
        )

        return completion.choices[0].message.content


@agent(method_name="generate_joke")
class PirateJokeAgent(JokeAgent):
    def generate_joke(self):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a funny sarcastic pirate"},
                {"role": "user", "content": "Tell me a joke about Donald Trump"},
            ],
        )

        return completion.choices[0].message.content


@workflow(name="jokes_generation")
def joke_generator():
    Traceloop.set_association_properties({"user_id": "user_12345"})

    print(f"Simple Joke: {JokeAgent().generate_joke()}")
    print(f"Pirate Joke: {PirateJokeAgent().generate_joke()}")


joke_generator()
