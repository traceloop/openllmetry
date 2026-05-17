import os
from ibm_watson_machine_learning.foundation_models import Model

from traceloop.sdk import Traceloop

from dotenv import load_dotenv
load_dotenv()

os.environ['OTEL_EXPORTER_OTLP_INSECURE'] = 'True'

Traceloop.init(app_name="watsonx_example")


def get_credentials(api_key):
    return {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": api_key,
    }


iam_api_key = os.environ["IAM_API_KEY"]
project_id = os.environ["PROJECT_ID"]

prompt_input = """Calculate result

Input:
what is the capital of China.

Output:
"""

model_id = "meta-llama/llama-2-70b-chat"
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 60,
    "min_new_tokens": 10,
    "random_seed": 111,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 2
}

model = Model(
    model_id=model_id,
    params=parameters,
    credentials=get_credentials(iam_api_key),
    project_id=project_id
    )

prompt_input = "What is the captical of China"
print(prompt_input)

generated_response = model.generate(prompt=prompt_input)
print(generated_response["results"][0]["generated_text"])
