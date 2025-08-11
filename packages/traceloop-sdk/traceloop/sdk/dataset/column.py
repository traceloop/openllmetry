from typing import Optional, TYPE_CHECKING

from .model import ColumnType
from traceloop.sdk.client.http import HTTPClient

if TYPE_CHECKING:
    from .dataset import Dataset


class Column:
    slug: str
    name: str
    type: ColumnType
    dataset_id: Optional[str] = None
    _http: HTTPClient
    _client: "Dataset"

    def __init__(
        self,
        http: HTTPClient,
        dataset: "Dataset",
        slug: str,
        name: str,
        type: ColumnType,
        dataset_id: str,
    ):
        self._http = http
        self._client = dataset
        self.slug = slug
        self.name = name
        self.type = type
        self.dataset_id = dataset_id

    def delete(self) -> None:
        """Remove this column from dataset"""
        if self._client is None:
            raise ValueError("Column must be associated with a dataset to delete")

        result = self._http.delete(f"datasets/{self._client.slug}/columns/{self.slug}")
        if result is None:
            raise Exception(f"Failed to delete column {self.slug}")

        self._client.columns.remove(self)

        # Update all rows by removing this column's values
        if self._client.rows:
            for row in self._client.rows:
                if self.slug in row.values:
                    del row.values[self.slug]

    def update(
        self, name: Optional[str] = None, type: Optional[ColumnType] = None
    ) -> None:
        """Update this column's properties"""
        update_data = {}
        if name is not None:
            update_data["name"] = name

        if type is not None:
            update_data["type"] = type

        if update_data:
            result = self._http.put(
                f"datasets/{self._client.slug}/columns/{self.slug}", update_data
            )
            if result is None:
                raise Exception(f"Failed to update column {self.slug}")

            if name is not None:
                self.name = name
            if type is not None:
                self.type = type
