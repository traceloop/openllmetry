from typing import Optional, Dict, Any, TYPE_CHECKING
from pydantic import PrivateAttr, Field

from .base import DatasetBaseModel

if TYPE_CHECKING:
    from .dataset import Dataset


class Row(DatasetBaseModel):
    id: str
    index: int = Field(alias="rowIndex")
    values: Dict[str, Any]
    dataset_id: str
    _client: Optional["Dataset"] = PrivateAttr(default=None)

    def delete(self) -> None:
        """Remove this row from dataset"""
        if self._client is None:
            from .dataset import Dataset
            self._client = Dataset()
        self._client.delete_row_api(self.dataset_id, self.id)

    def update(self, values: Dict[str, Any]) -> None:
        """Update this row's values"""
        if self._client is None:
            from .dataset import Dataset
            self._client = Dataset()
        self._client.update_cells_api(self.dataset_id, {self.id: values})
        self.values.update(values)

    def get_value(self, column_name: str) -> Any:
        """Get value by column name"""
        return self.values.get(column_name)

    def set_value(self, column_name: str, value: Any) -> None:
        """Set value by column name"""
        self.values[column_name] = value
        self.update({column_name: value})