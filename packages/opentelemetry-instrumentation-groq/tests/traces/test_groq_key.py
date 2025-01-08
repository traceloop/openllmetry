import os
import pytest
import requests

def test_groq_api_key():
    print("\nDebug - Entire os.environ:")
    for key, value in os.environ.items():
        masked_value = f"{value[:7]}..." if value else "None"
        print(f"{key}: {masked_value}")

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        pytest.fail("GROQ_API_KEY environment variable is not set or is empty")

    masked_key = f"{api_key[:7]}..." if len(api_key) > 7 else api_key
    print(f"\nDebug - API key starts with: {masked_key}")

    if not api_key.startswith("gsk_"):
        pytest.fail(f"API key should start with 'gsk_' but starts with: {api_key[:4]}")

    # Construct the full URL directly
    api_url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        print(f"Debug - API URL: {api_url}")  # Print the URL
        print(f"Debug - Headers: {headers}")  # Print the headers

        # Use requests library to make the POST request
        response = requests.post(
            api_url,
            headers=headers,
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 5
            },
            timeout=10
        )

        print(f"Debug - Response Status Code: {response.status_code}")
        print(f"Debug - Response Text: {response.text}")

        response.raise_for_status()  # Raise an exception for bad status codes

        # Process the response
        response_json = response.json()

        assert response_json["choices"][0]["message"]["content"] is not None, "Response content is missing"
        assert response_json["id"] is not None, "Response ID is missing"
        assert response_json["object"] == "chat.completion", "Incorrect response object type"
        assert response_json["usage"] and response_json["usage"]["total_tokens"] > 0, "Usage data is missing or invalid"

        print("\nAPI key verification successful!")

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])