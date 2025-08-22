import sys
import os

from traceloop.sdk.annotation.user_feedback import UserFeedback
from traceloop.sdk.datasets.datasets import Datasets
from traceloop.sdk.experiment.experiment import Experiment
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__
import httpx


class Client:
    """
    Traceloop Client for interacting with the Traceloop API.

    Applications should configure the client at startup time and continue to use it throughout the lifetime
    of the application, rather than creating instances on the fly. The best way to do this is with the
    singleton methods :func:`Traceloop.init()` and :func:`Traceloop.get()`. However, you may also call
    the constructor directly if you need to maintain multiple instances.
    """

    app_name: str
    api_endpoint: str
    api_key: str
    user_feedback: UserFeedback
    datasets: Datasets
    experiment: Experiment
    _http: HTTPClient
    _async_http: httpx.AsyncClient

    def __init__(
        self,
        api_key: str,
        app_name: str = sys.argv[0],
        api_endpoint: str = "https://api.traceloop.com",
    ):
        """
        Initialize a new Traceloop client.

        Args:
            api_key (str): Your Traceloop API key
            app_name (Optional[str], optional): The name of your application. Defaults to sys.argv[0].
            api_endpoint (Optional[str], optional): Custom API endpoint. Defaults to https://api.traceloop.com.
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key is required")

        self.app_name = app_name
        self.api_endpoint = api_endpoint or "https://api.traceloop.com"
        self.api_key = api_key
        self._http = HTTPClient(
            base_url=self.api_endpoint, api_key=self.api_key, version=__version__
        )
        self._async_http = httpx.AsyncClient(
            base_url=self.api_endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"traceloop-sdk/{__version__}",
            },
            timeout=httpx.Timeout(120.0),
        )
        self.user_feedback = UserFeedback(self._http, self.app_name)
        self.datasets = Datasets(self._http)
        experiment_slug = os.getenv("TRACELOOP_EXP_SLUG")
        self.experiment = Experiment(self._http, self._async_http, experiment_slug)
