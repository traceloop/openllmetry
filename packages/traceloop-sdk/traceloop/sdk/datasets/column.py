from typing import Optional, Dict, Any, TYPE_CHECKING
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
            from .dataset import Dataset
            self._client = Dataset()
        self._client.delete_column_api(self.dataset_id, self.id)

    def update(self, name: Optional[str] = None, type: Optional[ColumnType] = None) -> None:
        """Update this column's properties"""
        if self._client is None:
            from .dataset import Dataset
            self._client = Dataset()
        
        update_data = {}
        if name is not None:
            update_data["name"] = name
            self.name = name
            
        if type is not None:
            update_data["type"] = type
            self.type = type
            
        if update_data:
            self._client.update_column_api(self.dataset_id, self.id, **update_data)
            self.name = name
            self.type = type

