from typing import Optional, TYPE_CHECKING
from pydantic import PrivateAttr

from .base import DatasetBaseModel
from .model import ColumnType

if TYPE_CHECKING:
    from .dataset import Dataset


class Column(DatasetBaseModel):
    id: str
    name: str
    type: ColumnType
    dataset_id: str
    _client: Optional["Dataset"] = PrivateAttr(default=None)

    def delete(self) -> None:
        """Remove this column from dataset"""
        if self._client is None:
            raise ValueError("Column must be associated with a dataset to delete")

        self._client.delete_column_api(self.id)

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
            self.name = name
            self.type = type
