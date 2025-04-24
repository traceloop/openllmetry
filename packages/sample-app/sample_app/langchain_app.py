from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from traceloop.sdk import Traceloop

Traceloop.init(app_name="langchain_example")


def langchain_app():
    chat = ChatOpenAI(temperature=0)

    # Step 1: Get a joke about OpenTelemetry
    joke_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a funny sarcastic nerd."),
        HumanMessage(content="Tell me a joke about {subject}.")
    ])

    # Get the joke
    subject = "OpenTelemetry"
    joke_chain = joke_prompt | chat | StrOutputParser()
    joke = joke_chain.invoke({"subject": subject})

    print(f"Generated joke: {joke}")

    # Step 2: Translate the joke to Sindarin
    translation_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are an Elf."),
        HumanMessage(content=f"Translate this joke into Sindarin language:\n{joke}")
    ])

    # Get the translation
    translation_chain = translation_prompt | chat | StrOutputParser()
    translation = translation_chain.invoke({})

    result = {
        "subject": subject,
        "joke": joke,
        "text": translation
    }

    print(result)


langchain_app()
