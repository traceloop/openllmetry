import asyncio
import os
import requests
from openai import OpenAI

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import agent, workflow

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


Traceloop.init(app_name="joke_generation_service")


@agent(name="base_joke_generator", method_name="generate_joke")
class JokeAgent:
    async def generate_joke(self):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Tell me a joke about Donald Trump"}],
        )

        return completion.choices[0].message.content


@agent(method_name="generate_joke")
class PirateJokeAgent(JokeAgent):
    async def generate_joke(self):
        return await self.generation_helper()

    async def generation_helper(self):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a funny sarcastic pirate"},
                {"role": "user", "content": "Tell me a joke about Donald Trump"},
            ],
        )

        return completion.choices[0].message.content


@workflow(name="jokes_generation")
async def joke_generator():
    Traceloop.set_association_properties({"user_id": "user_12345"})

    requests.get("https://www.google.com")
    print(f"Simple Joke: {await JokeAgent().generate_joke()}")
    print(f"Pirate Joke: {await PirateJokeAgent().generate_joke()}")


asyncio.run(joke_generator())
