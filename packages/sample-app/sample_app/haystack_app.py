import os
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser
from haystack.pipelines import Pipeline
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init(app_name="haystack_app")

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

query = "OpenTelemetry"
result = pipeline.run(query)
print(result["answers"][0].answer)
