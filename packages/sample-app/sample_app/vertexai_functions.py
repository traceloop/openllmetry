import os
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, aworkflow

Traceloop.init(app_name="text_generation_service")

project_id = os.getenv('VERTEXAI_PROJECT_ID')
location = os.getenv('VERTEXAI_LOCATION')

vertexai.init(project=project_id, location=location)


@workflow("generate_content")
def generate_text() -> str:
    """Generate content with Multimodal Model (Gemini)"""

    multimodal_model = GenerativeModel("gemini-pro-vision")
    response = multimodal_model.generate_content(
        [
            Part.from_uri(
                "gs://generativeai-downloads/images/scones.jpg", mime_type="image/jpeg"
            ),
            "what is shown in this image?",
        ]
    )
    return response.text


@workflow("predict")
def predict_text() -> str:
    """Ideation example with a Large Language Model"""

    parameters = {
        "temperature": 0.1,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        "Give me ten interview questions for the role of program manager.",
        **parameters,
    )

    return response.text


@aworkflow("predict_async")
async def async_predict_text() -> str:
    """Async Ideation example with a Large Language Model"""

    parameters = {
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = await model.predict_async(
        "Give me ten interview questions for the role of program manager.",
        **parameters,
    )

    return response.text

if __name__ == "__main__":
    print(generate_text())
