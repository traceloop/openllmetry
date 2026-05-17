import boto3

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

Traceloop.init(app_name="joke_generation_service")
session = boto3.Session(profile_name="stg")
brt = session.client(service_name='bedrock-runtime')


@task(name="joke_creation")
def create_joke():

    response = brt.converse(
        modelId='us.amazon.nova-lite-v1:0',
        messages=[
            {
                "role": "user",
                "content": [{"text": "Tell me a joke about opentelemetry"}],
            }
        ],
        inferenceConfig={
            "maxTokens": 200,
            "temperature": 0.5,
        },
    )

    return response["output"]["message"]["content"][0]["text"]


@workflow(name="pirate_joke_generator")
def joke_workflow():
    print(create_joke())


joke_workflow()
