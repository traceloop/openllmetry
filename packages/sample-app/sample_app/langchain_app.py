import os
import openai

from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain, TransformChain

from traceloop.sdk import Traceloop

openai.api_key = os.getenv("OPENAI_API_KEY")
Traceloop.init(app_name="langchain_example")


def langchain_app():
    chat = ChatOpenAI(temperature=0)

    transform = TransformChain(
        input_variables=["subject"],
        output_variables=["prompt"],
        transform=lambda subject: {"prompt": f"Tell me a joke about {subject}."},
    )

    first_prompt_messages = [
        SystemMessage(content="You are a funny sarcastic nerd."),
        HumanMessage(content="{prompt}"),
    ]
    first_prompt_template = ChatPromptTemplate.from_messages(first_prompt_messages)
    first_chain = LLMChain(llm=chat, prompt=first_prompt_template, output_key="joke")

    second_prompt_messages = [
        SystemMessage(content="You are an Elf."),
        HumanMessagePromptTemplate.from_template(
            "Translate the joke below into Sindarin language:\n {joke}"
        ),
    ]
    second_prompt_template = ChatPromptTemplate.from_messages(second_prompt_messages)
    second_chain = LLMChain(llm=chat, prompt=second_prompt_template)

    workflow = SequentialChain(
        chains=[transform, first_chain, second_chain], input_variables=["subject"]
    )
    print(workflow({"subject": "OpenTelemetry"}))


langchain_app()
