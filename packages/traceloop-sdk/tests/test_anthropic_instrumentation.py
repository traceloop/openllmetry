from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from traceloop.sdk.decorators import workflow, task


def test_anthropic_completion(exporter):
    @task(name="joke_creation")
    def create_joke():
        client = Anthropic()
        response = client.completions.create(
            prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
            model="claude-instant-1.2",
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
        "anthropic.completion",
        "joke_creation.task",
        "pirate_joke_generator.workflow",
    ]
    anthropic_span = spans[0]
    assert (
        anthropic_span.attributes["llm.prompts.0.user"]
        == f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}"
    )
    assert anthropic_span.attributes.get("llm.completions.0.content")
