from typing import Dict, Any, Optional, List
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__
import os


class DatasetClient:
    """
    Singleton client for dataset API communication, following the PromptRegistryClient pattern
    """
    _http: HTTPClient

    def __new__(cls) -> "DatasetClient":
        if not hasattr(cls, "instance"):
            obj = cls.instance = super(DatasetClient, cls).__new__(cls)
            
            # Get API configuration from environment or defaults
            api_key = os.environ.get("TRACELOOP_API_KEY", "")
            api_endpoint = os.environ.get("TRACELOOP_BASE_URL", "https://api.traceloop.com")
            
            if not api_key:
                raise ValueError("TRACELOOP_API_KEY environment variable is required")
            
            obj._http = HTTPClient(base_url=api_endpoint, api_key=api_key, version=__version__)

        return cls.instance

    def create_dataset(self, name: str, description: Optional[str] = None, columns: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create new dataset"""
        print("NOMI - create_dataset")
        data = {"name": name}
        if description:
            data["description"] = description
        
        print(f"NOMI - data: {data}")
        result = self._http.post("projects/default/datasets", data)
        print(f"NOMI - result: {result}")
        if result is None:
            raise Exception("Failed to create dataset")
        return result

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve dataset by ID"""
        result = self._http.get(f"datasets/{dataset_id}")
        if result is None:
            raise Exception(f"Failed to get dataset {dataset_id}")
        return result

    def get_all_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets"""
        result = self._http.get("datasets")
        if result is None:
            raise Exception("Failed to get datasets")
        return result.get("datasets", [])

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete dataset"""
        # Assuming DELETE method exists in HTTPClient (would need to add)
        # For now, we'll use a POST with delete action
        result = self._http.post(f"datasets/{dataset_id}/delete", {})
        if result is None:
            raise Exception(f"Failed to delete dataset {dataset_id}")

    def add_column(self, dataset_id: str, name: str, col_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add column to dataset"""
        data = {
            "name": name,
            "type": col_type
        }
        if config:
            data["config"] = config
        
        result = self._http.post(f"datasets/{dataset_id}/columns", data)
        if result is None:
            raise Exception(f"Failed to add column to dataset {dataset_id}")
        return result

    def update_column(self, dataset_id: str, column_id: str, **kwargs) -> Dict[str, Any]:
        """Update column properties"""
        result = self._http.post(f"datasets/{dataset_id}/columns/{column_id}", kwargs)
        if result is None:
            raise Exception(f"Failed to update column {column_id}")
        return result

    def delete_column(self, dataset_id: str, column_id: str) -> None:
        """Delete column"""
        result = self._http.post(f"datasets/{dataset_id}/columns/{column_id}/delete", {})
        if result is None:
            raise Exception(f"Failed to delete column {column_id}")

    def add_row(self, dataset_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """Add single row to dataset"""
        data = {"values": values}
        result = self._http.post(f"datasets/{dataset_id}/rows", data)
        if result is None:
            raise Exception(f"Failed to add row to dataset {dataset_id}")
        return result

    def delete_row(self, dataset_id: str, row_id: str) -> None:
        """Delete row"""
        result = self._http.post(f"datasets/{dataset_id}/rows/{row_id}/delete", {})
        if result is None:
            raise Exception(f"Failed to delete row {row_id}")

    def update_cells(self, dataset_id: str, updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk update cells"""
        data = {"updates": updates}
        result = self._http.post(f"datasets/{dataset_id}/cells", data)
        if result is None:
            raise Exception(f"Failed to update cells in dataset {dataset_id}")
        return result

    def update_column_order(self, dataset_id: str, column_ids: List[str]) -> Dict[str, Any]:
        """Reorder columns"""
        data = {"column_ids": column_ids}
        result = self._http.post(f"datasets/{dataset_id}/column-order", data)
        if result is None:
            raise Exception(f"Failed to update column order in dataset {dataset_id}")
        return result