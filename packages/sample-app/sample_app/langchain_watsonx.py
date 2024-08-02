import os
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_ibm import WatsonxLLM

from traceloop.sdk import Traceloop

Traceloop.init(app_name="langchain_watsonx")

parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

watsonx_llm = WatsonxLLM(
    model_id="ibm/granite-13b-instruct-v2",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=os.environ.get("WATSONX_PROJECT_ID"),
    params=parameters,
)

template = "Generate a random question about {topic}: Question: "
prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=watsonx_llm)
print(llm_chain.invoke("dog"))
