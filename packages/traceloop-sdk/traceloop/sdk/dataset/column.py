from typing import Optional, TYPE_CHECKING

from .model import ColumnType
from .base import BaseDatasetEntity
from traceloop.sdk.client.http import HTTPClient

if TYPE_CHECKING:
    from .dataset import Dataset


class Column(BaseDatasetEntity):
    slug: str
    name: str
    type: ColumnType
    _dataset: "Dataset"
    deleted: bool = False

    def __init__(
        self,
        http: HTTPClient,
        dataset: "Dataset",
        slug: str,
        name: str,
        type: ColumnType,
    ):
        super().__init__(http)
        self._dataset = dataset
        self.slug = slug
        self.name = name
        self.type = type

    def delete(self) -> None:
        """Remove this column from dataset"""
        if self.deleted:
            raise Exception(f"Column {self.slug} already deleted")

        if self._dataset is None:
            raise ValueError("Column must be associated with a dataset to delete")

        result = self._http.delete(f"datasets/{self._dataset.slug}/columns/{self.slug}")
        if result is None:
            raise Exception(f"Failed to delete column {self.slug}")

        self._dataset.columns.remove(self)
        self.deleted = True

        # Update all rows by removing this column's values
        if self._dataset.rows:
            for row in self._dataset.rows:
                if self.slug in row.values:
                    del row.values[self.slug]

    def update(
        self, name: Optional[str] = None, type: Optional[ColumnType] = None
    ) -> None:
        """Update this column's properties"""
        if self.deleted:
            raise Exception(f"Column {self.slug} already deleted")

        update_data = {}
        if name is not None:
            update_data["name"] = name

        if type is not None:
            update_data["type"] = type

        if update_data:
            result = self._http.put(
                f"datasets/{self._dataset.slug}/columns/{self.slug}", update_data
            )
            if result is None:
                raise Exception(f"Failed to update column {self.slug}")

            if name is not None:
                self.name = name
            if type is not None:
                self.type = type
