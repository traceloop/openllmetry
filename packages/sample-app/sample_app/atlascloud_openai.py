import os

from dotenv import load_dotenv
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from traceloop.sdk import Traceloop

from sample_app.provider_factory import create_openai_compatible_client


def main():
    load_dotenv()

    if "TRACELOOP_METRICS_ENABLED" not in os.environ:
        os.environ["TRACELOOP_METRICS_ENABLED"] = "true"

    Traceloop.init(
        app_name="atlascloud-openai-demo",
        disable_batch=True,
        exporter=None if os.getenv("TRACELOOP_API_KEY") else ConsoleSpanExporter(),
    )

    config, client = create_openai_compatible_client("atlascloud")

    print(f"Provider: {config.name}")
    print(f"Base URL: {config.base_url}")
    print(f"Model: {config.default_model}")
    print("Sending request...")

    response = client.chat.completions.create(
        model=config.default_model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "请用一句话介绍 OpenLLMetry 的作用。"},
        ],
    )

    print("\n=== Response ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
