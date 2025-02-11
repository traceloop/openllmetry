from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain, TransformChain
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="langchain_example")

@workflow(name="langchain_nested_workflows")
def langchain_app():
    res = first_workflow()
    second_res = second_workflow(res)

    print(second_res)

    return second_res

def first_workflow():
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
        SystemMessage(content="You are an Israeli."),
        HumanMessagePromptTemplate.from_template(
            "Translate the joke below into Hebrew language:\n {joke}"
        ),
    ]
    second_prompt_template = ChatPromptTemplate.from_messages(second_prompt_messages)
    second_chain = LLMChain(llm=chat, prompt=second_prompt_template, output_key="translated_joke")

    workflow = SequentialChain(
        chains=[transform, first_chain, second_chain], input_variables=["subject"]
    )

    return workflow({"subject": "OpenTelemetry"})

def second_workflow(translated_joke):
    chat = ChatOpenAI(temperature=0)

    transform = TransformChain(
        input_variables=["translated_joke"],
        output_variables=["prompt"],
        transform=lambda translated_joke: {"prompt": f"transform the joke into a 5 sentence story {translated_joke}."},
    )

    first_prompt_messages = [
        SystemMessage(content="You are a funny story teller."),
        HumanMessage(content="{prompt}"),
    ]

    first_prompt_template = ChatPromptTemplate.from_messages(first_prompt_messages)
    first_chain = LLMChain(llm=chat, prompt=first_prompt_template, output_key="story")

    second_prompt_messages = [
        SystemMessage(content="You are an American-Israeli story teller"),
        HumanMessagePromptTemplate.from_template(
            "Translate the joke below into english language:\n {story}"
        ),
    ]

    second_prompt_template = ChatPromptTemplate.from_messages(second_prompt_messages)

    second_chain = LLMChain(llm=chat, prompt=second_prompt_template, output_key="translated_story")
    workflow = SequentialChain(
        chains=[transform, first_chain, second_chain], input_variables=["translated_joke"]
    )

    
    return workflow({"translated_joke": translated_joke})

langchain_app()
