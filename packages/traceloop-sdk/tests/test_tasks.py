import json
import pytest

from langchain_openai import ChatOpenAI
from traceloop.sdk.decorators import task
from opentelemetry.semconv.ai import SpanAttributes


@pytest.mark.vcr
def test_task_io_serialization_with_langchain(exporter):
    @task(name="answer_question")
    def answer_question():
        chat = ChatOpenAI(temperature=0)

        return chat.invoke("Is Berlin the capital of Germany? Answer with yes or no")

    answer_question()

    spans = exporter.get_finished_spans()

    assert [span.name for span in spans] == [
        'openai.chat',
        'answer_question.task',
    ]

    task_span = next(span for span in spans if span.name == 'answer_question.task')
    assert (json.loads(task_span.attributes.get(SpanAttributes.TRACELOOP_ENTITY_OUTPUT))["kwargs"]["content"] == "Yes")
