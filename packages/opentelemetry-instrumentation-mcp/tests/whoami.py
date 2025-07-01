from typing import Literal

from mcp.types import Request, RequestParams, Result
from pydantic import RootModel


class WhoamiRequest(Request[RequestParams | None, Literal["whoami"]]):
    method: Literal["whoami"]
    params: RequestParams | None = None


class WhoamiResult(Result):
    name: str


class TestServerRequest(RootModel[WhoamiRequest]):
    pass


class TestClientResult(RootModel[WhoamiResult]):
    pass
