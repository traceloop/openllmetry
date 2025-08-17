from typing import Dict, Any, TYPE_CHECKING

from .base import BaseDatasetEntity
from traceloop.sdk.client.http import HTTPClient

if TYPE_CHECKING:
    from .dataset import Dataset


class Row(BaseDatasetEntity):
    id: str
    values: Dict[str, Any]
    _dataset: "Dataset"
    deleted: bool = False

    def __init__(
        self,
        http: HTTPClient,
        dataset: "Dataset",
        id: str,
        values: Dict[str, Any],
    ):
        super().__init__(http)
        self._dataset = dataset
        self.id = id
        self.values = values

    def delete(self) -> None:
        """Remove this row from dataset"""
        if self.deleted:
            raise Exception(f"Row {self.id} already deleted")

        result = self._http.delete(f"datasets/{self._dataset.slug}/rows/{self.id}")
        if result is None:
            raise Exception(f"Failed to delete row {self.id}")
        if self._dataset.rows and self in self._dataset.rows:
            self._dataset.rows.remove(self)
            self.deleted = True

    def update(self, values: Dict[str, Any]) -> None:
        """Update this row's values"""
        if self.deleted:
            raise Exception(f"Row {self.id} already deleted")

        data = {"values": values}
        result = self._http.put(f"datasets/{self._dataset.slug}/rows/{self.id}", data)
        if result is None:
            raise Exception(f"Failed to update row {self.id}")
        self.values.update(values)
