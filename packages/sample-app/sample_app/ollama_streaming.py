from traceloop.sdk import Traceloop
from ollama import chat

Traceloop.init()

base_model = "gemma3:1b"


def ollama_chat():
    stream_response = chat(
        model=base_model,
        messages=[
            {
                'role': 'user',
                'content': 'Tell a joke about opentelemetry'
            },
        ],
        stream=True
    )

    full_response = ""
    print("Streaming response:")
    for chunk in stream_response:
        if chunk.message and chunk.message.content:
            print(chunk.message.content, end="", flush=True)
            full_response += chunk.message.content


def main():
    ollama_chat()


if __name__ == "__main__":
    main()
