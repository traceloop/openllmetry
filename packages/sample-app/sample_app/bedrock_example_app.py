import boto3
import json

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="joke_generation_service")
brt = boto3.client(service_name='bedrock-runtime')


# def fullname(o):
#     klass = o.__class__
#     module = klass.__module__
#     if module == 'builtins':
#         return klass.__qualname__ # avoid outputs like 'builtins.str'
#     return module + '.' + klass.__qualname__

@task(name="joke_creation")
def create_joke():

    body = json.dumps({
        "prompt": "Tell me a joke about opentelemetry",
        "max_tokens": 200,
        "temperature": 0.5,
        "p": 0.5,
    })

    response = brt.invoke_model(
        body=body, 
        modelId='cohere.command-text-v14', 
        accept='application/json', 
        contentType='application/json'
    )
    # response_body = json.loads(response.get('body').read())

    # return response_body.get('generations')[0].get('text')
    return None

@workflow(name="pirate_joke_generator")
def joke_workflow():
    print(create_joke())

joke_workflow()