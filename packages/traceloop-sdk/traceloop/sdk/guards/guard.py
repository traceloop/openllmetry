from ..client.http import HTTPClient


class Guard:
    """
    Guard class for creating guards in Traceloop.
    """

    _http: HTTPClient
    _app_name: str

    def __init__(self, http: HTTPClient, app_name: str):
        self._http = http
        self._app_name = app_name

    def from_evaluator(self, slug: str):

        pass