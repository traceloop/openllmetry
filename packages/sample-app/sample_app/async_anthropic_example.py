import asyncio
import requests
from anthropic import AsyncAnthropic

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import agent, workflow

anthropic = AsyncAnthropic()

Traceloop.init(app_name="joke_generation_service")


@agent(name="base_joke_generator", method_name="generate_joke")
class JokeAgent:
    async def generate_joke(self):
        response = await anthropic.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Tell me a joke about Donald Trump"}],
            model="claude-3-haiku-20240307",
        )
        return response.content[0].text


@agent(method_name="generate_joke")
class PirateJokeAgent(JokeAgent):
    async def generate_joke(self):
        return await self.generation_helper()

    async def generation_helper(self):
        response = await anthropic.messages.create(
            max_tokens=1024,
            system="You are a funny sarcastic pirate",
            messages=[
                {"role": "user", "content": "Tell me a joke about Donald Trump"},
            ],
            model="claude-3-haiku-20240307",
        )

        return response.content[0].text


@workflow(name="jokes_generation")
async def joke_generator():
    Traceloop.set_association_properties({"user_id": "user_12345"})
    requests.get("https://www.google.com")
    print(f"Simple Joke: {await JokeAgent().generate_joke()}")
    print(f"Pirate Joke: {await PirateJokeAgent().generate_joke()}")

    await asyncio.sleep(1)

asyncio.run(joke_generator())
