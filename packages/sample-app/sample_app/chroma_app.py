# Based on https://cookbook.openai.com/examples/vector_databases/chroma/hyde-with-chroma-and-openai

import os
import pandas as pd
import openai
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init(app_name="chroma_app")


openai.api_key = os.getenv("OPENAI_API_KEY")
embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"))

claim_df = pd.read_json("data/scifact/scifact_claims.jsonl", lines=True)
corpus_df = pd.read_json("data/scifact/scifact_corpus.jsonl", lines=True)

chroma_client = chromadb.Client()
scifact_corpus_collection = chroma_client.create_collection(
    name="scifact_corpus", embedding_function=embedding_function
)

batch_size = 100

for i in range(0, len(corpus_df), batch_size):
    batch_df = corpus_df[i: i + batch_size]
    scifact_corpus_collection.add(
        ids=batch_df["doc_id"]
        .apply(lambda x: str(x))
        .tolist(),  # Chroma takes string IDs.
        documents=(
            batch_df["title"] + ". " + batch_df["abstract"].apply(lambda x: " ".join(x))
        ).to_list(),  # We concatenate the title and abstract.
        metadatas=[
            {"structured": structured}
            for structured in batch_df["structured"].to_list()
        ],  # We also store the metadata, though we don't use it in this example.
    )


def build_prompt_with_context(claim, context):
    return [
        {
            "role": "system",
            "content": "I will ask you to assess whether a particular scientific claim, based on evidence provided. "
            + "Output only the text 'True' if the claim is true, 'False' if the claim is false, or 'NEE' if there's "
            + "not enough evidence.",
        },
        {
            "role": "user",
            "content": f""""
The evidence is the following:

{' '.join(context)}

Assess the following claim on the basis of the evidence. Output only the text 'True' if the claim is true,
'False' if the claim is false, or 'NEE' if there's not enough evidence. Do not output any other text.

Claim:
{claim}

Assessment:
""",
        },
    ]


@workflow("assess_claims")
def assess_claims(claims):
    claim_query_result = scifact_corpus_collection.query(
        query_texts=claims, include=["documents", "distances"], n_results=3
    )
    responses = []
    # Query the OpenAI API
    for claim, context in zip(claims, claim_query_result["documents"]):
        # If no evidence is provided, return NEE
        if len(context) == 0:
            responses.append("NEE")
            continue
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=build_prompt_with_context(claim=claim, context=context),
            max_tokens=3,
        )
        # Strip any punctuation or whitespace from the response
        formatted_response = response.choices[0].message.content.strip("., ")
        print("Claim: ", claim)
        print("Response: ", formatted_response)
        responses.append(formatted_response)

    return responses


samples = claim_df.sample(2)
assess_claims(samples["claim"].tolist())
