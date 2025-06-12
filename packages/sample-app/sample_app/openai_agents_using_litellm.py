import asyncio
import os
from agents import Agent, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents import ModelSettings
from traceloop.sdk import Traceloop

Traceloop.init()

useModel = LitellmModel(
    model="groq/llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)


agent = Agent(
    name="GroqAgent",
    instructions="You are a helpful assistant that answers all questions",
    model=useModel,
    model_settings=ModelSettings(
        temperature=0.3, max_tokens=1024, top_p=0.2, frequency_penalty=0.3
    ),
)


async def main():

    result = await Runner.run(
        starting_agent=agent,
        input="What is quantum computing?",
    )
    summary = result.final_output
    print("\n Output:\n", summary)


if __name__ == "__main__":
    asyncio.run(main())
