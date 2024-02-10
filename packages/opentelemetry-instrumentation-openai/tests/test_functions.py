def test_open_ai_function_calls(exporter, openai_client):
    openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
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

    spans = exporter.get_finished_spans()
    open_ai_span = spans[0]
    assert (
        open_ai_span.attributes["llm.prompts.0.content"]
        == "What's the weather like in Boston?"
    )
    assert (
        open_ai_span.attributes["llm.request.functions.0.name"] == "get_current_weather"
    )
    assert (
        open_ai_span.attributes["llm.request.functions.0.description"]
        == "Get the current weather in a given location"
    )
    assert (
        open_ai_span.attributes["llm.completions.0.function_call.name"]
        == "get_current_weather"
    )
