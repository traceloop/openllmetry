from typing import Optional, TYPE_CHECKING
from pydantic import PrivateAttr

from .model import ColumnType, DatasetBaseModel
from traceloop.sdk.client.http import HTTPClient

if TYPE_CHECKING:
    from .dataset import Dataset


class Column(DatasetBaseModel):
    id: str
    name: str
    type: ColumnType
    dataset_id: str
    _http: HTTPClient
    _client: Optional["Dataset"] = PrivateAttr(default=None)

    def __init__(self, http: HTTPClient):
        self._http = http

    def delete(self) -> None:
        """Remove this column from dataset"""
        if self._client is None:
            raise ValueError("Column must be associated with a dataset to delete")
        
        result = self._http.delete(
            f"projects/default/datasets/{self._client.slug}/columns/{self.id}"
        )
        if result is None:
            raise Exception(f"Failed to delete column {self.id}")

        self._client.columns.remove(self)

        # Update all rows by removing this column's values
        for row in self._client.rows:
            if self.id in row.values:
                del row.values[self.id]

    def update(self, name: Optional[str] = None, type: Optional[ColumnType] = None) -> None:
        """Update this column's properties"""
        if self._client is None:
            raise ValueError("Column must be associated with a dataset to update")

        update_data = {}
        if name is not None:
            update_data["name"] = name

        if type is not None:
            update_data["type"] = type

        if update_data:
            self._client.update_column_api(column_id=self.id, data=update_data)
            if name is not None:
                self.name = name
            if type is not None:
                self.type = type
