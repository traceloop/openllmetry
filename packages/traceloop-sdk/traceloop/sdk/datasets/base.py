from abc import ABC
from datetime import datetime
from traceloop.sdk.client.http import HTTPClient


class BaseDatasetEntity(ABC):
    """
    Abstract base class for dataset-related objects with common attributes
    """

    created_at: datetime
    updated_at: datetime
    _http: HTTPClient

    def __init__(
        self,
        http: HTTPClient,
        created_at: datetime = datetime.now(),
        updated_at: datetime = datetime.now(),
    ):
        self._http = http
        self.created_at = created_at
        self.updated_at = updated_at
