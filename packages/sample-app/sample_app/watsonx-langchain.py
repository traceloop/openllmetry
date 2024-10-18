import os
import types
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.foundation_models import ModelInference
from pprint import pprint
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from langchain_ibm import WatsonxLLM

from dotenv import load_dotenv
load_dotenv()

os.environ['OTEL_EXPORTER_OTLP_INSECURE'] = 'True'

Traceloop.init(app_name="watsonx_llm_langchain_question")


def watsonx_llm_init() -> ModelInference:

    watsonx_llm_parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 100,
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 1,
        GenTextParamsMetaNames.TEMPERATURE: 0.5,
        GenTextParamsMetaNames.TOP_K: 50,
        GenTextParamsMetaNames.TOP_P: 1,
    }

    watsonx_llm = WatsonxLLM(
        model_id="ibm/granite-13b-instruct-v2",
        url="https://us-south.ml.cloud.ibm.com",
        apikey=os.getenv("IAM_API_KEY"),
        project_id=os.getenv("PROJECT_ID"),
        params=watsonx_llm_parameters,
    )

    return watsonx_llm


@workflow(name="watsonx_llm_langchain_question")
def watsonx_llm_generate(question):
    watsonx_llm = watsonx_llm_init()
    return watsonx_llm.invoke(question)


question_simple = "What is 1 + 1?"

response = watsonx_llm_generate(question_simple)

if isinstance(response, types.GeneratorType):
    for chunk in response:
        print(chunk, end='')

pprint(response)
