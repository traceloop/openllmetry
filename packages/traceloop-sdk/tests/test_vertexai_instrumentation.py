import os
import pytest
import asyncio
import vertexai
from traceloop.sdk.decorators import workflow, aworkflow
from vertexai.language_models import TextGenerationModel
from vertexai.preview.generative_models import GenerativeModel, Part

project_id = os.getenv('VERTEXAI_PROJECT_ID')
location = os.getenv('VERTEXAI_LOCATION')

# Initialize Vertex AI
vertexai.init(project=project_id, location=location)

def test_vertexai_generate_content(exporter):
    @workflow("generate_content")
    def generate_text() -> str:
        """Generate content with Multimodal Model (Gemini)"""

        # Load the model
        multimodal_model = GenerativeModel("gemini-pro-vision")
        # Query the model
        response = multimodal_model.generate_content(
            [
                # Add an example image
                Part.from_uri(
                    "gs://generativeai-downloads/images/scones.jpg", mime_type="image/jpeg"
                ),
                # Add an example query
                "what is shown in this image?",
            ]
        )
        return response.text

    generate_text()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.generate_content",
        "generate_content.workflow",
    ]

def test_vertexai_predict(exporter):
    @workflow("predict")
    def predict_text() -> str:
        """Ideation example with a Large Language Model"""

        parameters = {
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = model.predict(
            "Give me ten interview questions for the role of program manager.",
            **parameters,
        )

        return response.text
    
    predict_text()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict",
        "predict.workflow",
    ]
def test_vertexai_predict_async(exporter):
    @aworkflow("predict_async")
    async def async_predict_text() -> str:
        """Ideation example with a Large Language Model"""

        parameters = {
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        model = TextGenerationModel.from_pretrained("text-bison@001")
        response = await model.predict_async(
            "Give me ten interview questions for the role of program manager.",
            **parameters,
        )

        return response.text
    
    asyncio.run(async_predict_text())

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict_async",
        "predict_async.workflow",
    ]

def test_vertexai_stream(exporter):
    @workflow("stream_prediction")
    def streaming_prediction() -> str:
        """Streaming Text Example with a Large Language Model"""

        text_generation_model = TextGenerationModel.from_pretrained("text-bison")
        parameters = {
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        responses = text_generation_model.predict_streaming(prompt="Give me ten interview questions for the role of program manager.", **parameters)
        
        result = [response for response in responses]
        return result
    
    streaming_prediction()

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict_streaming",
        "stream_prediction.workflow",
    ]

def test_vertexai_stream_async(exporter):
    @aworkflow("stream_prediction_async")
    async def async_streaming_prediction() -> str:
        """Streaming Text Example with a Large Language Model"""

        text_generation_model = TextGenerationModel.from_pretrained("text-bison")
        parameters = {
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        responses = text_generation_model.predict_streaming(prompt="Give me ten interview questions for the role of program manager.", **parameters)
        
        result = [response for response in responses]
        return result
    
    asyncio.run(async_streaming_prediction())

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "vertexai.predict_streaming_async",
        "stream_prediction_async.workflow",
    ]