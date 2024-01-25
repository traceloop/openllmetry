import os
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.pipelines import Pipeline


def test_haystack(exporter):
    prompt = PromptTemplate(
        prompt="Tell me a joke about {query}\n",
        output_parser=AnswerParser(),
    )

    prompt_node = PromptNode(
        model_name_or_path="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        default_prompt_template=prompt,
    )

    pipeline = Pipeline()
    pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Query"])
    pipeline.run("OpenTelemetry")

    spans = exporter.get_finished_spans()
    assert set(
        [
            "openai.chat",
            "PromptNode.task",
            "haystack_pipeline.workflow",
        ]
    ).issubset([span.name for span in spans])
