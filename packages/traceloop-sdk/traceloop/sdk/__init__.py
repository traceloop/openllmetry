import os
import requests
from typing import Optional

from colorama import Fore
from opentelemetry.util.re import parse_env_headers

from traceloop.sdk.tracing.tracing import TracerWrapper, set_correlation_id

TRACELOOP_API_ENDPOINT = "https://api.traceloop.dev"


class Traceloop:
    __tracer_wrapper: TracerWrapper

    @staticmethod
    def init(app_name: Optional[str] = None) -> None:
        api_key = os.getenv("TRACELOOP_API_KEY")
        api_endpoint = os.getenv("TRACELOOP_API_ENDPOINT") or TRACELOOP_API_ENDPOINT
        headers = os.getenv("TRACELOOP_HEADERS") or {}
        if isinstance(headers, str):
            headers = parse_env_headers(headers)

        # auto-create a dashboard on Traceloop if no export endpoint is provided
        if api_endpoint == TRACELOOP_API_ENDPOINT and not api_key:
            if os.path.exists("/tmp/traceloop_key.txt"):
                api_key = open("/tmp/traceloop_key.txt").read()
            else:
                print(
                    Fore.YELLOW
                    + "No Traceloop API key provided, auto-creating a dashboard on Traceloop",
                )
                res = requests.post(
                    "https://app.traceloop.com/api/registration/auto-create"
                )
                api_key = res.json()["apiKey"]
                print(Fore.YELLOW + "Set TRACELOOP_API_KEY to", api_key)
                open("/tmp/traceloop_key.txt", "w").write(api_key)
            print(
                Fore.GREEN
                + f"\nGo to https://app.traceloop.com/trace?orgToken={api_key} to see a live dashboard\n",
            )
            print(Fore.RESET)

        if api_key:
            headers = {
                "Authorization": f"Bearer {api_key}",
            } | headers

        TracerWrapper.set_endpoint(api_endpoint, headers)
        Traceloop.__tracer_wrapper = TracerWrapper()

    @staticmethod
    def set_correlation_id(correlation_id: str) -> None:
        set_correlation_id(correlation_id)
