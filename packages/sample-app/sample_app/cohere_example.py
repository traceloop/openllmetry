import os
import cohere
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init()

co = cohere.Client(os.environ.get("COHERE_API_KEY"))


@workflow(name="pirate_joke_generator")
def joke_workflow():
    return co.chat(model="command", message="Tell me a joke, pirate style")


@workflow(name="rerank")
def rerank():
    query = "What is the capital of the United States?"
    documents = [
        "Carson City is the capital city of the American state of Nevada."
        + " At the  2010 United States Census, Carson City had a population of 55,274.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands"
        + " in the Pacific Ocean that are a political division controlled by the "
        + "United States. Its capital is Saipan.",
        "Charlotte Amalie is the capital and largest city of the United States "
        + "Virgin Islands. It has about 20,000 people. The city is on the island of Saint Thomas.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially "
        + "as the District of Columbia) is the capital of the United States. It is a federal district. ",
        "Capital punishment (the death penalty) has existed in the United States "
        + "since before the United States was a country. As of 2017, capital "
        + "punishment is legal in 30 of the 50 states.",
        "North Dakota is a state in the United States. 672,591 people lived"
        + " in North Dakota in the year 2010. The capital and seat of government is Bismarck.",
    ]

    results = co.rerank(
        query=query,
        documents=documents,
        top_n=3,
        model="rerank-multilingual-v2.0",
    )
    print(results)


# res = joke_workflow()
# print(res)
rerank()
