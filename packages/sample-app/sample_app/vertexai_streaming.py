import os
import asyncio
import vertexai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, aworkflow
from vertexai.language_models import TextGenerationModel

Traceloop.init(app_name="stream_prediction_service")

project_id = os.getenv('VERTEXAI_PROJECT_ID')
location = os.getenv('VERTEXAI_LOCATION')

# Initialize Vertex AI
vertexai.init(project=project_id, location=location)

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

@aworkflow("stream_prediction_async")
async def async_streaming_prediction() -> str:
    """Async Streaming Text Example with a Large Language Model"""

    text_generation_model = TextGenerationModel.from_pretrained("text-bison")
    parameters = {
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    responses = text_generation_model.predict_streaming_async(prompt="Give me ten interview questions for the role of program manager.", **parameters)

    result = [response async for response in responses]
    return result


if __name__ == "__main__":
    # print(streaming_prediction())
    print(asyncio.run(async_streaming_prediction()))
