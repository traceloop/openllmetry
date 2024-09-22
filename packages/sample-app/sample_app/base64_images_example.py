import base64
import os
import requests


def upload_base64_image(image_path, message_index, message_content_index, image_format, trace_id, span_id, api_key):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {
        "message_index": message_index,
        "message_content_index": message_content_index,
        "image_data": encoded_image,
        "image_format": image_format
    }

    url = f"http://localhost:3000/traces/{trace_id}/spans/{span_id}/images"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print(f"Image {image_name} uploaded successfully.")
    else:
        print(f"Failed to upload image {image_name}. Status code: {response.status_code}")


def get_image_urls(trace_id, span_id, api_key):
    url = f"http://localhost:3000/traces/{trace_id}/spans/{span_id}/images"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)

    print(response.json())

    if response.status_code == 200:
        image_url = response.json().get('url')
        return image_url
    else:
        print(f"Failed to retrieve image URL. Status code: {response.status_code}")
        return None


if __name__ == "__main__":
    image_path = "/Users/galklm/Downloads/itamar.jpeg"
    image_name = "itamar.jpeg"
    trace_id = "tid_156"
    span_id = "sid_156"
    api_key = os.environ.get("TRACELOOP_API_KEY")

    image_url = get_image_urls(trace_id, span_id, api_key)
    if image_url:
        print(f"Retrieved image URL: {image_url}")
