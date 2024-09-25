import vertexai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from vertexai.generative_models import GenerativeModel

Traceloop.init(app_name="stream_prediction_service")

vertexai.init()


@workflow("stream_prediction")
def streaming_prediction() -> str:
    """Streaming Text Example with a Large Language Model"""

    model = GenerativeModel(
        model_name="gemini-1.5-flash-001",
        system_instruction=[
            "You are a helpful language translator.",
            "Your mission is to translate text in English to French.",
        ],
    )

    prompt = """
    User input: I like bagels.
    Answer:
    """

    contents = [prompt]

    response = model.generate_content(contents)

    return response.text


if __name__ == "__main__":
    print(streaming_prediction())
