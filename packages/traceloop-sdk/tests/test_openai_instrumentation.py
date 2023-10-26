import openai
from traceloop.sdk.decorators import workflow, task


def test_simple_workflow(exporter):
    @task(name="joke_creation")
    def create_joke():
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Tell me a joke about opentelemetry"}
            ],
        )
        return completion.choices[0].message.content

    @workflow(name="pirate_joke_generator")
    def joke_workflow():
        create_joke()

    joke_workflow()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "openai.chat",
        "joke_creation.task",
        "pirate_joke_generator.workflow",
    ]
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["llm.prompts.0.content"]
        == "Tell me a joke about opentelemetry"
    )
    assert open_ai_span.attributes.get("llm.completions.0.content")


def test_open_ai_function_calls(exporter):
    @task(name="function_call_test")
    def function_call_test():
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "What's the weather like in Boston?"}
            ],
            functions=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call="auto",
        )
        return completion.choices[0].message.content

    @workflow(name="function_call_test_workflow")
    def function_call_test_workflow():
        function_call_test()

    function_call_test_workflow()

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["llm.completions.0.function_call.name"]
        == "get_current_weather"
    )
