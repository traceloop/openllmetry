import boto3
import json
import pytest

def test_bedrock_access():
    """
    Tests access to Amazon Bedrock and invokes a test model.
    """
    try:
        # Create a Bedrock client
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name="ap-south-1"  # Make sure this is your correct region
        )

        # Define a simple request body (for Titan Text G1 - Express)
        body = json.dumps({
            "inputText": "Hello Bedrock, can you generate some text for me?",
            "textGenerationConfig": {
                "maxTokenCount": 50,
                "temperature": 0.7,
                "topP": 0.9
            }
        })

        # Invoke the model
        response = bedrock_runtime.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=body,
            contentType="application/json",
            accept="application/json"
        )

        # Check for a successful response
        response_body = json.loads(response.get("body").read())
        assert response_body.get("results", [])  # Basic check for response structure

        print("Successfully invoked Amazon Bedrock model!")

    except Exception as e:
        print(f"Error during Bedrock model invocation: {e}")
        pytest.fail(f"Failed to invoke Bedrock model. Error: {e}")