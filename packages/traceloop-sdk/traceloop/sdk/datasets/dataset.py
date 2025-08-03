import csv
from datetime import datetime
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
from pydantic import Field, PrivateAttr
import os

from .base import DatasetBaseModel, ColumnType
from .column import Column
from .row import Row
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__

if TYPE_CHECKING:
    try:
        import pandas as pd
    except ImportError:
        pd = None


class Dataset(DatasetBaseModel):
    """
    Combined Dataset class with HTTP client functionality for dataset API communication
    """
    id: Optional[str] = None
    name: str
    slug: str
    description: Optional[str] = None
    columns: List[Column] = Field(default_factory=list)
    rows: List[Row] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    _http: Optional[HTTPClient] = PrivateAttr(default=None)
    
    def _get_http_client(self) -> HTTPClient:
        """Get or create HTTP client instance"""
        if self._http is None:
            # Get API configuration from environment or defaults
            api_key = os.environ.get("TRACELOOP_API_KEY", "")
            api_endpoint = os.environ.get("TRACELOOP_BASE_URL", "https://api.traceloop.com")
            
            if not api_key:
                raise ValueError("TRACELOOP_API_KEY environment variable is required")
            
            self._http = HTTPClient(base_url=api_endpoint, api_key=api_key, version=__version__)
        
        return self._http

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> "Dataset":
        """Class method to create dataset from CSV file"""
        print("NOMI - from_csv")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        if name is None:
            name = path.stem
        
        # Read CSV and infer column types
        columns = []
        rows_data = []
        
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # Detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            # Create columns based on CSV headers
            for i, field_name in enumerate(reader.fieldnames):
                columns.append(Column(
                    id=f"col_{i}",
                    name=field_name,
                    type=ColumnType.STRING,  # Default to STRING, could be enhanced with type inference
                    dataset_id="temp"  # Will be set after dataset creation
                ))
            
            # Read all rows
            for row_idx, row_data in enumerate(reader):
                rows_data.append({
                    "id": f"row_{row_idx}",
                    "index": row_idx,
                    "values": dict(row_data),
                    "dataset_id": "temp"  # Will be set after dataset creation
                })

                print(f"NOMI - row_data: {row_data}")
                print(f"NOMI - columns: {columns}")
        
        # Create dataset instance
        dataset = cls(
            name=name,
            slug=slug,
            description=description,
            columns=columns,
        )
        
        # Create dataset via API and get ID
        api_dataset = dataset.create_dataset_api(name, description)
        dataset.id = api_dataset["id"]
        dataset.created_at = datetime.fromisoformat(api_dataset["created_at"])
        dataset.updated_at = datetime.fromisoformat(api_dataset["updated_at"])
        
        # Update column and row dataset_ids
        for col in dataset.columns:
            col.dataset_id = dataset.id
            col._client = dataset
        
        # Create columns via API
        for col in dataset.columns:
            api_col = dataset.add_column_api(dataset.id, col.name, col.type, col.config)
            col.id = api_col["id"]
        
        # Create rows
        for row_data in rows_data:
            row_data["dataset_id"] = dataset.id
            row = Row(**row_data)
            row._client = dataset
            dataset.rows.append(row)
            
            # Add row via API
            dataset.add_row_api(dataset.id, row.values)
        return dataset

    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> "Dataset":
        """Class method to create dataset from pandas DataFrame"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for from_dataframe(). Install with: pip install pandas")
        
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected pandas DataFrame")
        
        if name is None:
            name = f"Dataset from DataFrame"
        
        # Create columns from DataFrame
        columns = []
        for i, col_name in enumerate(df.columns):
            # Infer column type from pandas dtype
            dtype = df[col_name].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                col_type = ColumnType.NUMBER
            elif pd.api.types.is_bool_dtype(dtype):
                col_type = ColumnType.BOOLEAN
            else:
                col_type = ColumnType.STRING
            
            columns.append(Column(
                id=f"col_{i}",
                name=str(col_name),
                type=col_type,
                dataset_id="temp"  # Will be set after dataset creation
            ))
        
        # Create dataset instance
        dataset = cls(
            name=name,
            slug=slug,
            description=description,
            columns=columns,
        )
        
        # Create dataset via API and get ID
        api_dataset = dataset.create_dataset_api(name, description)
        dataset.id = api_dataset["id"]
        dataset.created_at = datetime.fromisoformat(api_dataset["created_at"])
        dataset.updated_at = datetime.fromisoformat(api_dataset["updated_at"])
        
        # Update column dataset_ids and create via API
        for col in dataset.columns:
            col.dataset_id = dataset.id
            col._client = dataset
            api_col = dataset.add_column_api(dataset.id, col.name, col.type, col.config)
            col.id = api_col["id"]
        
        # Create rows from DataFrame
        for row_idx, (_, pandas_row) in enumerate(df.iterrows()):
            # Convert pandas row to dict, handling NaN values
            row_values = {}
            for col_name in df.columns:
                value = pandas_row[col_name]
                if pd.isna(value):
                    row_values[col_name] = None
                else:
                    row_values[col_name] = value
            
            row = Row(
                id=f"row_{row_idx}",
                index=row_idx,
                values=row_values,
                dataset_id=dataset.id,
                _client=dataset
            )
            dataset.rows.append(row)
            
            # Add row via API
            dataset.add_row_api(dataset.id, row.values)
        return dataset

    def add_column(self, name: str, col_type: ColumnType, config: Optional[Dict[str, Any]] = None) -> Column:
        """Add new column (returns Column object)"""
        api_col = self.add_column_api(self.id, name, col_type, config)
        
        column = Column(
            id=api_col["id"],
            name=name,
            type=col_type,
            config=config,
            dataset_id=self.id,
            _client=self
        )
        self.columns.append(column)
        return column
    
    # API Methods (formerly from DatasetClient)
    
    def create_dataset_api(self, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create new dataset"""
        print("NOMI - create_dataset")
        data = {"name": name}
        if description:
            data["description"] = description
        
        print(f"NOMI - data: {data}")
        result = self._get_http_client().post("projects/default/datasets", data)
        print(f"NOMI - result: {result}")
        if result is None:
            raise Exception("Failed to create dataset")
        return result

    def get_dataset_api(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve dataset by ID"""
        result = self._get_http_client().get(f"datasets/{dataset_id}")
        if result is None:
            raise Exception(f"Failed to get dataset {dataset_id}")
        return result

    def get_all_datasets_api(self) -> List[Dict[str, Any]]:
        """List all datasets"""
        result = self._get_http_client().get("datasets")
        if result is None:
            raise Exception("Failed to get datasets")
        return result.get("datasets", [])

    def delete_dataset_api(self, dataset_id: str) -> None:
        """Delete dataset"""
        # Assuming DELETE method exists in HTTPClient (would need to add)
        # For now, we'll use a POST with delete action
        result = self._get_http_client().post(f"datasets/{dataset_id}/delete", {})
        if result is None:
            raise Exception(f"Failed to delete dataset {dataset_id}")

    def add_column_api(self, dataset_id: str, name: str, col_type: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add column to dataset"""
        data = {
            "name": name,
            "type": col_type
        }
        if config:
            data["config"] = config
        
        result = self._get_http_client().post(f"datasets/{dataset_id}/columns", data)
        if result is None:
            raise Exception(f"Failed to add column to dataset {dataset_id}")
        return result

    def update_column_api(self, dataset_id: str, column_id: str, **kwargs) -> Dict[str, Any]:
        """Update column properties"""
        result = self._get_http_client().post(f"datasets/{dataset_id}/columns/{column_id}", kwargs)
        if result is None:
            raise Exception(f"Failed to update column {column_id}")
        return result

    def delete_column_api(self, dataset_id: str, column_id: str) -> None:
        """Delete column"""
        result = self._get_http_client().post(f"datasets/{dataset_id}/columns/{column_id}/delete", {})
        if result is None:
            raise Exception(f"Failed to delete column {column_id}")

    def add_row_api(self, dataset_id: str, values: Dict[str, Any]) -> Dict[str, Any]:
        """Add single row to dataset"""
        data = {"values": values}
        result = self._get_http_client().post(f"datasets/{dataset_id}/rows", data)
        if result is None:
            raise Exception(f"Failed to add row to dataset {dataset_id}")
        return result

    def delete_row_api(self, dataset_id: str, row_id: str) -> None:
        """Delete row"""
        result = self._get_http_client().post(f"datasets/{dataset_id}/rows/{row_id}/delete", {})
        if result is None:
            raise Exception(f"Failed to delete row {row_id}")

    def update_cells_api(self, dataset_id: str, updates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk update cells"""
        data = {"updates": updates}
        result = self._get_http_client().post(f"datasets/{dataset_id}/cells", data)
        if result is None:
            raise Exception(f"Failed to update cells in dataset {dataset_id}")
        return result

    def update_column_order_api(self, dataset_id: str, column_ids: List[str]) -> Dict[str, Any]:
        """Reorder columns"""
        data = {"column_ids": column_ids}
        result = self._get_http_client().post(f"datasets/{dataset_id}/column-order", data)
        if result is None:
            raise Exception(f"Failed to update column order in dataset {dataset_id}")
        return result