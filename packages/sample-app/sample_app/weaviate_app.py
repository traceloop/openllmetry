# Tested with weaviate-client==4.4.0
# This is not yet compatible with traceloop
import os
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
import weaviate
import weaviate.classes as wvc
from weaviate.util import generate_uuid5

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow

# Init trace
Traceloop.init(
    app_name="weaviate_app",
    disable_batch=True,
    exporter=ConsoleSpanExporter()
)

# Code is adapted from official documentation.
# V4 documentation: https://weaviate.io/developers/weaviate/client-libraries/python

client = weaviate.connect_to_wcs(
    cluster_url=os.getenv("WEAVIATE_CLUSTER_URL"),
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]
    }
)

COLLECTION_NAME = "JeopardyQuestion"


@task("get_collection")
def get_collection(client):
    try:
        collection = client.collections.get(COLLECTION_NAME)
        return collection
    except weaviate.exceptions.UnexpectedStatusCodeError:
        return None


@task("create_collection")
def create_collection(client):
    collection = client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
        generative_config=wvc.config.Configure.Generative.openai(),
        properties=[
            wvc.config.Property(
                name="title",
                data_type=wvc.config.DataType.TEXT
            )
        ]
    )
    print("Created collection: ", collection)


@task("insert_data")
def insert_data(collection):
    print("Inserting data... ", end="")
    new_uuid = collection.data.insert(
        properties={
            "question": "This is the capital of Australia."
        }
    )
    print("OK - UUID: ", new_uuid)

    data_objects = list()
    for i in range(5):
        properties = {"question": f"Test Question {i+1}"}
        data_object = wvc.data.DataObject(
            properties=properties,
            uuid=generate_uuid5(properties)
        )
        data_objects.append(data_object)


@task("query_collection")
def query_collection(collection, query):
    response = collection.query.bm25(
        query=query,
        limit=1,
    )
    return response.objects


@workflow("example")
def example_workflow():
    collection = get_collection(client)

    # Apparently Weaviate response does not implement a __len__,
    # therefore we have to check for a None
    if collection is None:
        create_collection(client)

    result = query_collection(collection, "Australia")

    if not result:
        insert_data(collection)
        result = query_collection(collection, "Australia")

    # Should return 'This is the capital of Australia'
    print("Retrieved response", result[0])


example_workflow()
client.close()
