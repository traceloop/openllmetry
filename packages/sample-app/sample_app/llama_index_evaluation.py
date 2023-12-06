import os
import nest_asyncio
import asyncio
import openai

from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    Response,
)
from llama_index.llms import OpenAI
from llama_index.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from llama_index.evaluation import DatasetGenerator
from llama_index.evaluation import BatchEvalRunner

import pandas as pd
from traceloop.sdk import Traceloop

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

nest_asyncio.apply()

openai.api_key = os.environ["OPENAI_API_KEY"]

pd.set_option("display.max_colwidth", 0)

# gpt-4
gpt4 = OpenAI(temperature=0, model="gpt-4")
service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)
relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)
correctness_gpt4 = CorrectnessEvaluator(service_context=service_context_gpt4)

documents = SimpleDirectoryReader("data/paul_graham").load_data()

# create vector index
llm = OpenAI(temperature=0.3, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

dataset_generator = DatasetGenerator.from_documents(
    documents, service_context=service_context
)

questions = dataset_generator.generate_questions_from_nodes(num=1)
print(questions)

# Traceloop.init(app_name="llama_index_example")

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=1,
)


async def main():
    eval_results = await runner.aevaluate_queries(
        vector_index.as_query_engine(), queries=questions
    )
    print(eval_results)


if __name__ == "__main__":
    asyncio.run(main())
