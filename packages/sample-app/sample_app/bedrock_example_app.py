import boto3
import json

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="joke_generation_service")
brt = boto3.client(service_name='bedrock-runtime')


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

    response_body = json.loads(response.get('body').read())

    return response_body.get('generations')[0].get('text')


@workflow(name="pirate_joke_generator")
def joke_workflow():
    print(create_joke())


joke_workflow()
