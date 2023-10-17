import os
import requests

from typing import Optional
from colorama import Fore
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.util.re import parse_env_headers

from traceloop.sdk.config import (
    base_url,
    is_prompt_registry_enabled,
    is_tracing_enabled,
)
from traceloop.sdk.prompts.client import PromptRegistryClient
from traceloop.sdk.tracing.tracing import TracerWrapper, set_correlation_id


class Traceloop:
    __tracer_wrapper: TracerWrapper

    @staticmethod
    def init(
        app_name: Optional[str] = None,
        api_endpoint: str = base_url(),
        api_key: str = None,
        headers: dict[str, str] = {},
        disable_batch=False,
        exporter: SpanExporter = None,
    ) -> None:
        if is_prompt_registry_enabled():
            PromptRegistryClient().run()

        if not is_tracing_enabled():
            print(Fore.YELLOW + "Traceloop is disabled")
            return

        if exporter:
            print(Fore.GREEN + "Traceloop exporting traces to a custom exporter")

        api_key = os.getenv("TRACELOOP_API_KEY") or api_key
        api_endpoint = os.getenv("TRACELOOP_BASE_URL") or api_endpoint
        headers = os.getenv("TRACELOOP_HEADERS") or headers
        if isinstance(headers, str):
            headers = parse_env_headers(headers)

        # auto-create a dashboard on Traceloop if no export endpoint is provided
        if not exporter and api_endpoint == "https://api.traceloop.com" and not api_key:
            headers = None  # disable headers if we're auto-creating a dashboard
            if os.path.exists("/tmp/traceloop_key.txt") and os.path.exists(
                "/tmp/traceloop_url.txt"
            ):
                api_key = open("/tmp/traceloop_key.txt").read()
                access_url = open("/tmp/traceloop_url.txt").read()
            else:
                print(
                    Fore.YELLOW
                    + "No Traceloop API key provided, auto-creating a dashboard on Traceloop",
                )
                res = requests.post(
                    "https://app.traceloop.com/api/registration/auto-create"
                ).json()
                access_url = f"https://app.traceloop.com/trace?skt={res['uiAccessKey']}"
                api_key = res["apiKey"]
                print(Fore.YELLOW + "TRACELOOP_API_KEY=", api_key)
                open("/tmp/traceloop_key.txt", "w").write(api_key)
                open("/tmp/traceloop_url.txt", "w").write(access_url)
            print(
                Fore.GREEN + f"\nGo to {access_url} to see a live dashboard\n",
            )

        if headers:
            print(
                Fore.GREEN
                + f"Traceloop exporting traces to {api_endpoint}, authenticating with custom headers"
            )

        if api_key and not headers:
            print(
                Fore.GREEN
                + f"Traceloop exporting traces to {api_endpoint} authenticating with bearer token"
            )
            headers = {
                "Authorization": f"Bearer {api_key}",
            }

        print(Fore.RESET)

        TracerWrapper.set_endpoint(api_endpoint, headers)
        Traceloop.__tracer_wrapper = TracerWrapper(
            disable_batch=disable_batch, exporter=exporter
        )

    @staticmethod
    def set_correlation_id(correlation_id: str) -> None:
        set_correlation_id(correlation_id)
