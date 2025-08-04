from typing import Optional, Dict, Any, TYPE_CHECKING
from pydantic import PrivateAttr, Field

from .base import DatasetBaseModel

if TYPE_CHECKING:
    from .dataset import Dataset


class Row(DatasetBaseModel):
    id: str
    row_index: int = Field(alias="rowIndex")
    values: Dict[str, Any]
    dataset_id: str
    _client: Optional["Dataset"] = PrivateAttr(default=None)

    def delete(self) -> None:
        """Remove this row from dataset"""
        if self._client is None:
            raise ValueError("Row must be associated with a dataset to delete")
        self._client.delete_row(self.id)
        self._client.rows.remove(self)

    def update(self, values: Dict[str, Any]) -> None:
        """Update this row's values"""
        if self._client is None:
            raise ValueError("Row must be associated with a dataset to update")
        self._client.update_row_api(self.id, values)
        self.values.update(values)
