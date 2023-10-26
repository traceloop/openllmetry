import os
import sys
from deprecated import deprecated
import requests

from typing import Optional
from colorama import Fore
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.util.re import parse_env_headers

from traceloop.sdk.config import (
    is_content_tracing_enabled,
    is_tracing_enabled,
)
from traceloop.sdk.fetcher import Fetcher
from traceloop.sdk.prompts.client import PromptRegistryClient
from traceloop.sdk.tracing.content_allow_list import ContentAllowList
from traceloop.sdk.tracing.tracing import (
    TracerWrapper,
    set_association_properties,
    set_correlation_id,
)


class Traceloop:
    __tracer_wrapper: TracerWrapper

    @staticmethod
    def init(
        app_name: Optional[str] = sys.argv[0],
        api_endpoint: str = "https://api.traceloop.com",
        api_key: str = None,
        headers: dict[str, str] = {},
        disable_batch=False,
        exporter: SpanExporter = None,
        traceloop_sync_enabled: bool = True,
    ) -> None:
        api_endpoint = os.getenv("TRACELOOP_BASE_URL") or api_endpoint

        if traceloop_sync_enabled and api_endpoint.find("traceloop.com") != -1:
            Fetcher(
                base_url=api_endpoint,
                prompt_registry=PromptRegistryClient()._registry,
                content_allow_list=ContentAllowList(),
            ).run()
        else:
            print(
                Fore.YELLOW + "Tracloop syncing configuration and prompts" + Fore.RESET
            )

        if not is_tracing_enabled():
            print(Fore.YELLOW + "Tracing is disabled" + Fore.RESET)
            return

        enable_content_tracing = is_content_tracing_enabled()

        if exporter:
            print(Fore.GREEN + "Traceloop exporting traces to a custom exporter")

        api_key = os.getenv("TRACELOOP_API_KEY") or api_key
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

        TracerWrapper.set_static_params(
            app_name, enable_content_tracing, api_endpoint, headers
        )
        Traceloop.__tracer_wrapper = TracerWrapper(
            disable_batch=disable_batch, exporter=exporter
        )

    @staticmethod
    @deprecated(version="0.0.62", reason="Use set_association_properties instead")
    def set_correlation_id(correlation_id: str) -> None:
        set_correlation_id(correlation_id)

    def set_association_properties(properties: dict) -> None:
        set_association_properties(properties)
