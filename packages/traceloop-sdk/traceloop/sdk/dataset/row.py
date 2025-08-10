from typing import Optional, Dict, Any, TYPE_CHECKING
from pydantic import PrivateAttr

from .model import DatasetBaseModel
from traceloop.sdk.client.http import HTTPClient

if TYPE_CHECKING:
    from .dataset import Dataset


class Row(DatasetBaseModel):
    id: str
    values: Dict[str, Any]
    dataset_id: str
    _dataset: "Dataset"
    _http: HTTPClient

    def __init__(self, http: HTTPClient, dataset: "Dataset", id: str, values: Dict[str, Any], dataset_id: str):
        self._http = http
        self._dataset = dataset
        self.id = id
        self.values = values
        self.dataset_id = dataset_id

    def delete(self) -> None:
        """Remove this row from dataset"""
        result = self._http.delete(f"projects/default/datasets/{self.slug}/rows/{self.id}")
        if result is None:
            raise Exception(f"Failed to delete row {self.id}")
        self._dataset.rows.remove(self)

    def update(self, values: Dict[str, Any]) -> None:
        """Update this row's values"""
        data = {"values": values}
        result = self._http.put(
            f"projects/default/datasets/{self.slug}/rows/{self.id}",
            data
        )
        if result is None:
            raise Exception(f"Failed to update row {self.id}")
        self.values.update(values)
