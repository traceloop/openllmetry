from google.cloud import aiplatform
from traceloop.sdk.decorators import workflow, task

aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used is not set
    project='my-project',

    # the Vertex AI region you will use
    # defaults to us-central1
    location='us-central1',

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket='gs://my_staging_bucket',

    # custom google.auth.credentials.Credentials
    # environment default credentials used if not set
    credentials=my_credentials,

    # customer managed encryption key resource name
    # will be applied to all Vertex AI resources if set
    encryption_spec_key_name=my_encryption_key_name,

    # the name of the experiment to use to track
    # logged metrics and parameters
    experiment='my-experiment',

    # description of the experiment above
    experiment_description='my experiment description'
)


def test_vertexai_completion(exporter):
    @task(name="joke_creation")
    def create_joke():
        client = Anthropic()
        response = client.completions.create(
            prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
            model="gemini-pro",
            max_tokens_to_sample=2048,
            top_p=0.1,
        )
        print(response)
        return response

    @workflow(name="pirate_joke_generator")
    def joke_workflow():
        create_joke()

    joke_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.completion",
        "joke_creation.task",
        "pirate_joke_generator.workflow",
    ]
