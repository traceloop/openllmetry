# traceloop-sdk

Traceloop’s Python SDK allows you to easily start monitoring and debugging your LLM execution. Tracing is done in a non-intrusive way, built on top of OpenTelemetry. You can choose to export the traces to Traceloop, or to your existing observability stack.


## Installation


You can now install only the integrations you need, or groups of them:

- All LLM providers:
    ```bash
    pip install traceloop-sdk[llm]
    ```
- All frameworks:
    ```bash
    pip install traceloop-sdk[frameworks]
    ```
- All vector stores:
    ```bash
    pip install traceloop-sdk[vectorstores]
    ```
- All cloud providers:
    ```bash
    pip install traceloop-sdk[cloud]
    ```
- Everything:
    ```bash
    pip install traceloop-sdk[all]
    ```
- Or any combination, e.g.:
    ```bash
    pip install traceloop-sdk[openai,chromadb]
    ```

This keeps your install minimal and fast!

---

```python
Traceloop.init(app_name="joke_generation_service")

@workflow(name="joke_creation")
def create_joke():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Tell me a joke about opentelemetry"}],
    )

    return completion.choices[0].message.content
```
