from typing import Optional, Dict, Any, TYPE_CHECKING
from pydantic import PrivateAttr

from .base import DatasetBaseModel, ColumnType

if TYPE_CHECKING:
    from .client import DatasetClient


class Column(DatasetBaseModel):
    id: str
    name: str
    type: ColumnType
    config: Optional[Dict[str, Any]] = None
    dataset_id: str
    _client: Optional["DatasetClient"] = PrivateAttr(default=None)

    def delete(self) -> None:
        """Remove this column from dataset"""
        if self._client is None:
            from .client import DatasetClient
            self._client = DatasetClient()
        self._client.delete_column(self.dataset_id, self.id)

    def update(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        """Update this column's properties"""
        if self._client is None:
            from .client import DatasetClient
            self._client = DatasetClient()
        
        update_data = {}
        if name is not None:
            update_data["name"] = name
            self.name = name
        if config is not None:
            update_data["config"] = config
            self.config = config
            
        if update_data:
            self._client.update_column(self.dataset_id, self.id, **update_data)