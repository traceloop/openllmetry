import os
import openai


from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from traceloop.sdk.prompts import render_prompt_by_key

openai.api_key = os.getenv("OPENAI_API_KEY")
Traceloop.init(app_name="prompt_registry_example_app")

@task(name="some_task")
def trigger_prompt():
    prompt_args = render_prompt_by_key("bla", var="king", var2="sexy")
    completion = openai.ChatCompletion.create(**prompt_args)

    return completion.choices[0].message.content

@workflow(name="prompt_registry_example_workflow")
def test_prompt_registry():
    print(trigger_prompt())

test_prompt_registry()