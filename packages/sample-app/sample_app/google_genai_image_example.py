#!/usr/bin/env python3
"""
Sample app to test Google Generative AI with image support
"""

import os
from dotenv import load_dotenv
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

load_dotenv()

# Initialize Traceloop SDK
Traceloop.init(app_name="google_genai_image_example")

try:
    # Use the new Google GenAI SDK
    import google.genai as genai
    from google.genai import types

    print("=== Google GenAI Image Example with Traceloop ===")

    # Initialize client
    client = genai.Client(vertexai=True, project=os.environ.get("GOOGLE_CLOUD_PROJECT"), location="global")

    @workflow("describe_image_google_genai")
    def describe_image_from_local_path(image_path: str) -> str:
        """Describe an image using Google GenAI from a local file path"""

        # Load the image from local path
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        # Determine mime type from file extension
        if image_path.lower().endswith(('.png', '.PNG')):
            mime_type = "image/png"
        elif image_path.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
            mime_type = "image/jpeg"
        elif image_path.lower().endswith(('.gif', '.GIF')):
            mime_type = "image/gif"
        elif image_path.lower().endswith(('.webp', '.WEBP')):
            mime_type = "image/webp"
        else:
            mime_type = "image/jpeg"  # Default fallback

        # Create image part using the new SDK structure
        image_part = types.Part(
            inline_data=types.Blob(
                mime_type=mime_type,
                data=image_data
            )
        )

        print(f"Created image part with mime_type: {image_part.inline_data.mime_type}")
        print(f"Image data size: {len(image_part.inline_data.data)} bytes")

        # Create contents with text and image
        contents = [
            "Describe what you see in this image in detail.",
            image_part
        ]

        try:
            # Make API call
            response = client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=contents
            )

            print(f"Response: {response.text}")
            return response.text

        except Exception as e:
            print(f"API call failed: {e}")
            return f"Error: {str(e)}"

    # Test with the elephant image
    image_path = os.path.join(os.path.dirname(__file__), "..", "data", "vision", "elephant.jpeg")

    if os.path.exists(image_path):
        print(f"\n📸 Testing with image: {image_path}")
        result = describe_image_from_local_path(image_path)
        print(f"\n🤖 Description: {result}")
    else:
        print(f"\n⚠️ Image not found at: {image_path}")
        print("Please ensure the elephant.jpeg file exists in data/vision/")

except ImportError as e:
    print(f"Google GenAI SDK not available: {e}")
    print("Please install: pip install google-genai")

print("\n✅ Done testing Google GenAI image support with Traceloop!")
