import os
import dotenv
import vertexai
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow
from vertexai.generative_models import GenerativeModel, Part

dotenv.load_dotenv()

Traceloop.init(app_name="gemini_vision_example")

# Initialize Vertex AI
# You can set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables
# or pass them directly to vertexai.init()
vertexai.init()


@workflow("describe_image_from_path")
def describe_image_from_local_path(image_path: str) -> str:
    """Describe an image using Gemini model from a local file path"""

    model = GenerativeModel("gemini-2.5-flash")

    # Load the image from local path
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Create image part from bytes
    image_part = Part.from_data(
        data=image_data,
        mime_type="image/jpeg"
    )

    # Generate content with image and text prompt
    response = model.generate_content([
        "Describe what you see in this image in detail. What is the main subject and what are they doing?",
        image_part
    ])

    return response.text


@workflow("analyze_image_from_gcs")
def analyze_image_from_gcs(gcs_uri: str) -> str:
    """Analyze an image using Gemini model from Google Cloud Storage URI"""

    model = GenerativeModel("gemini-2.5-flash")

    # Create image part from GCS URI
    image_part = Part.from_uri(
        uri=gcs_uri,
        mime_type="image/jpeg"
    )

    # Generate content with image and text prompt
    response = model.generate_content([
        "What objects do you see in this image? List them and describe their characteristics.",
        image_part
    ])

    return response.text


@workflow("multi_turn_vision_chat")
def multi_turn_vision_chat(image_path: str) -> str:
    """Have a multi-turn conversation about an image"""

    model = GenerativeModel("gemini-2.5-flash")

    # Load the image from local path
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Create image part from bytes
    image_part = Part.from_data(
        data=image_data,
        mime_type="image/jpeg"
    )

    # Start a chat session
    chat = model.start_chat()

    # First turn with image
    response1 = chat.send_message([
        "What animal do you see in this image?",
        image_part
    ])

    # Follow-up questions without image (continues the conversation)
    response2 = chat.send_message("What is the natural habitat of this animal?")
    response3 = chat.send_message("What are some interesting facts about this animal?")

    # Return the final response
    return f"First: {response1.text}\nSecond: {response2.text}\nThird: {response3.text}"


if __name__ == "__main__":
    # Path to the sample image
    image_path = "data/vision/elephant.jpeg"

    # Check if the image exists
    if os.path.exists(image_path):
        print("=== Describing Image from Local Path ===")
        result1 = describe_image_from_local_path(image_path)
        print(result1)

        print("\n=== Multi-turn Vision Chat ===")
        result2 = multi_turn_vision_chat(image_path)
        print(result2)
    else:
        print(f"Image not found at {image_path}")

    # Example with GCS URI (uncomment if you have a GCS image)
    # print("\n=== Analyzing Image from GCS ===")
    # gcs_result = analyze_image_from_gcs("gs://your-bucket/your-image.jpg")
    # print(gcs_result)
