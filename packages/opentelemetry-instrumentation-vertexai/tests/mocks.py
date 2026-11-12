"""Offline mocks for the VertexAI instrumentation tests.

vcr.py cannot record gRPC traffic (see issue #417), so instead of cassettes
the tests patch the SDK methods that would otherwise make real gRPC calls and
feed them canned responses built from the real SDK response types. The SDK is
initialized with anonymous credentials and a dummy project, so no test ever
touches the network or needs GCP credentials.
"""

import vertexai
from google.auth.credentials import AnonymousCredentials
from vertexai.generative_models import GenerationResponse
from vertexai.language_models import TextGenerationResponse
from vertexai.language_models._language_models import (
    MultiCandidateTextGenerationResponse,
)

GEMINI_RESPONSE_TEXT = "This is a mocked model response."

GEMINI_RESPONSE_DICT = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"text": GEMINI_RESPONSE_TEXT}],
            },
            "finish_reason": 1,  # FinishReason.STOP
        }
    ],
    "usage_metadata": {
        "prompt_token_count": 10,
        "candidates_token_count": 5,
        "total_token_count": 15,
    },
}

BISON_RESPONSE_TEXT = "mocked model response"
BISON_STREAM_CHUNKS = ("mocked ", "model ", "response")


def init_vertexai():
    """Initialize the SDK without real GCP credentials or network access."""
    vertexai.init(project="test-project", credentials=AnonymousCredentials())


def build_predict_response(text=BISON_RESPONSE_TEXT):
    """A MultiCandidateTextGenerationResponse like TextGenerationModel.predict returns."""
    return MultiCandidateTextGenerationResponse(
        text=text,
        _prediction_response=None,
        candidates=[TextGenerationResponse(text=text, _prediction_response=None)],
    )


def build_chat_response(text=BISON_RESPONSE_TEXT):
    """A TextGenerationResponse like ChatSession.send_message returns."""
    return TextGenerationResponse(text=text, _prediction_response=None)


def build_stream_response(chunks=BISON_STREAM_CHUNKS):
    """An iterator of TextGenerationResponse chunks like the streaming methods return."""
    return iter(
        TextGenerationResponse(text=chunk, _prediction_response=None) for chunk in chunks
    )


def offline_generate_content(self, *args, **kwargs):
    return GenerationResponse.from_dict(GEMINI_RESPONSE_DICT)


def offline_from_pretrained(cls, model_name):
    """Local replacement for ``ModelGardenModel.from_pretrained`` (no auth check)."""
    return cls(model_id=model_name)


def offline_predict(self, *args, **kwargs):
    return build_predict_response()


async def offline_predict_async(self, *args, **kwargs):
    return build_predict_response()


def offline_predict_streaming(self, *args, **kwargs):
    return build_stream_response()


async def offline_predict_streaming_async(self, *args, **kwargs):
    for chunk in build_stream_response():
        yield chunk


def offline_send_message(self, *args, **kwargs):
    return build_chat_response()


def offline_send_message_streaming(self, *args, **kwargs):
    return build_stream_response()
