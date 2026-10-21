"""Offline mocks for Vertex AI SDK calls (no GCP credentials or network required)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from google.cloud.aiplatform import models as aiplatform_models
from vertexai.generative_models import GenerationResponse
from vertexai.language_models import ChatModel, TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel

BISON_COMPLETION = (
    "1. Tell me about a time you led a cross-functional project.\n"
    "2. How do you handle conflicting priorities?\n"
)
BISON_STREAM_CHUNKS = ["1. Tell me ", "about a time ", "you led a project."]
CHAT_COMPLETION = "There are eight planets in the solar system."
CHAT_STREAM_CHUNKS = ["There are ", "eight planets ", "in the solar system."]
GEMINI_COMPLETION = "A plate of scones on a table."
GEMINI_USAGE = SimpleNamespace(
    total_token_count=100,
    prompt_token_count=60,
    candidates_token_count=40,
)


class FakeGeminiCandidate:
    def __init__(self, text: str, index: int = 0, finish_reason_value: int = 1):
        self.text = text
        self.index = index
        self.finish_reason = SimpleNamespace(value=finish_reason_value)


class FakeGeminiResponse(GenerationResponse):
    """Minimal GenerationResponse for offline tests and event emission."""

    def __init__(self, text: str = GEMINI_COMPLETION, usage: SimpleNamespace = GEMINI_USAGE):
        self._text = text
        self._usage = usage
        self._candidates = [FakeGeminiCandidate(text)]
        self._raw_response = SimpleNamespace(usage_metadata=usage)

    @property
    def text(self) -> str:
        return self._text

    @property
    def candidates(self):
        return self._candidates

    @property
    def usage_metadata(self):
        return self._usage


def _make_bison_prediction_dict(content: str) -> dict:
    return {
        "content": content,
        "safetyAttributes": {
            "blocked": False,
            "errors": [],
            "categories": [],
            "scores": [],
        },
    }


def _make_chat_prediction_dict(content: str) -> dict:
    return {
        "candidates": [{"content": content}],
        "safetyAttributes": [
            {
                "blocked": False,
                "errors": [],
                "categories": [],
                "scores": [],
            }
        ],
    }


def make_bison_prediction(content: str) -> aiplatform_models.Prediction:
    return aiplatform_models.Prediction(
        predictions=[_make_bison_prediction_dict(content)],
        deployed_model_id="",
    )


def make_chat_prediction(content: str) -> aiplatform_models.Prediction:
    return aiplatform_models.Prediction(
        predictions=[_make_chat_prediction_dict(content)],
        deployed_model_id="",
    )


def make_text_generation_model(model_id: str) -> TextGenerationModel:
    model = object.__new__(TextGenerationModel)
    model._model_id = model_id
    model._endpoint = MagicMock()
    model._endpoint_name = (
        f"projects/test/locations/us-central1/publishers/google/models/{model_id}"
    )
    model._endpoint._prediction_client = MagicMock()
    model._endpoint._prediction_async_client = MagicMock()
    return model


def make_chat_model(model_id: str) -> ChatModel:
    model = object.__new__(ChatModel)
    model._model_id = model_id
    model._endpoint = MagicMock()
    model._endpoint_name = (
        f"projects/test/locations/us-central1/publishers/google/models/{model_id}"
    )
    model._endpoint._prediction_client = MagicMock()
    return model


def _fake_generative_model_init(self, model_name, **_kwargs):
    self._model_name = model_name


def patch_text_generation_from_pretrained():
    def _from_pretrained(cls, model_name, **_kwargs):
        return make_text_generation_model(model_name)

    return patch.object(TextGenerationModel, "from_pretrained", classmethod(_from_pretrained))


def patch_chat_from_pretrained():
    def _from_pretrained(cls, model_name, **_kwargs):
        return make_chat_model(model_name)

    return patch.object(ChatModel, "from_pretrained", classmethod(_from_pretrained))


def configure_predict(model: TextGenerationModel, content: str = BISON_COMPLETION):
    model._endpoint.predict.return_value = make_bison_prediction(content)


def configure_predict_async(model: TextGenerationModel, content: str = BISON_COMPLETION):
    model._endpoint.predict_async = AsyncMock(
        return_value=make_bison_prediction(content)
    )


def configure_chat_predict(model: ChatModel, content: str = CHAT_COMPLETION):
    model._endpoint.predict.return_value = make_chat_prediction(content)


def patch_bison_streaming(chunks: list[str]):
    dicts = [_make_bison_prediction_dict(chunk) for chunk in chunks]
    return patch(
        "vertexai.language_models._language_models._streaming_prediction"
        ".predict_stream_of_dicts_from_single_dict",
        return_value=iter(dicts),
    )


def patch_bison_streaming_async(chunks: list[str]):
    dicts = [_make_bison_prediction_dict(chunk) for chunk in chunks]

    async def _async_stream(**_kwargs):
        for item in dicts:
            yield item

    return patch(
        "vertexai.language_models._language_models._streaming_prediction"
        ".predict_stream_of_dicts_from_single_dict_async",
        new=_async_stream,
    )


def patch_chat_streaming(chunks: list[str]):
    dicts = [_make_chat_prediction_dict(chunk) for chunk in chunks]
    return patch(
        "vertexai.language_models._language_models._streaming_prediction"
        ".predict_stream_of_dicts_from_single_dict",
        return_value=iter(dicts),
    )


def patch_gemini_generate_content(response=None):
    if response is None:
        response = FakeGeminiResponse()
    return patch.multiple(
        GenerativeModel,
        __init__=_fake_generative_model_init,
        _generate_content=MagicMock(return_value=response),
    )
