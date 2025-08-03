import csv
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import Field, PrivateAttr
import pandas as pd 
import os

from traceloop.sdk.datasets.model import ColumnDefinition, ValuesMap, CreateDatasetRequest, CreateDatasetResponse, CreateRowsResponse, ColumnType
from .base import DatasetBaseModel
from .column import Column
from .row import Row
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__


class Dataset(DatasetBaseModel):
    """
    Dataset class dataset API communication
    """
    id: Optional[str] = None
    name: str
    slug: str
    description: Optional[str] = None
    columns_definition: List[ColumnDefinition] = Field(default_factory=list)
    columns: List[Column] = Field(default_factory=list)
    rows: List[Row] = Field(default_factory=list)
    last_version: Optional[str] = None
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
    
    def _convert_rows_by_names_to_col_ids(self, rows_with_names: List[ValuesMap]) -> List[ValuesMap]:
        """Convert multiple rows from column names to column IDs"""
        return [
            {column.id: val for column, val in zip(self.columns, row_data.values())}
            for row_data in rows_with_names
        ]
    
    def _create_columns(self, api_dataset: CreateDatasetResponse):
            """Create Column objects from API response which includes column IDs"""
            for column_id, column_def in api_dataset.columns.items():
                column = Column(
                    id=column_id,
                    name=column_def.name,
                    type=column_def.type,
                    dataset_id=self.id,
                    _client=self
                )
                self.columns.append(column)
    
    def _create_rows(self, rows_response: CreateRowsResponse):
        for idx, row_obj in enumerate(rows_response.rows):
            row = Row(
                id=row_obj.id,
                index=idx,
                values=row_obj.values,
                dataset_id=self.id,
                _client=self
            )
            self.rows.append(row)

    @classmethod
    def from_csv(
        cls,
        file_path: str,
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> "Dataset":
        """Class method to create dataset from CSV file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        columns_definition: List[ColumnDefinition] = []
        rows_with_names: List[ValuesMap] = []
        
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            # Detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            
            reader = csv.DictReader(csvfile, delimiter=delimiter)
            
            for _, field_name in enumerate(reader.fieldnames):
                columns_definition.append(ColumnDefinition(
                    name=field_name,
                    type=ColumnType.STRING,
                ))
            
            for _, row_data in enumerate(reader):
                rows_with_names.append(dict(row_data))
        
        # Create dataset instance
        dataset = cls(
            name=name,
            slug=slug,
            description=description,
            columns_definition=columns_definition,
        )
        
        dataset_response = dataset.create_dataset(CreateDatasetRequest(slug=slug, name=name, description=description, columns=columns_definition))
        
        dataset._create_columns(dataset_response)
        
        rows_with_ids = dataset._convert_rows_by_names_to_col_ids(rows_with_names)
        
        rows_response = dataset.add_rows(dataset_response.slug, rows_with_ids)
        
        dataset._create_rows(rows_response)
        
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
        # Create column definitions from DataFrame
        columns_definition: List[ColumnDefinition] = []
        for col_name in df.columns:
            dtype = df[col_name].dtype
            if pd.api.types.is_bool_dtype(dtype):
                col_type = ColumnType.BOOLEAN
            elif pd.api.types.is_numeric_dtype(dtype):
                col_type = ColumnType.NUMBER
            else:
                col_type = ColumnType.STRING
            
            columns_definition.append(ColumnDefinition(
                name=col_name,
                type=col_type
            ))
        
        # Create dataset instance
        dataset = cls(
            name=name,
            slug=slug,
            description=description,
            columns_definition=columns_definition,
        )
        
        dataset_response = dataset.create_dataset(CreateDatasetRequest(slug=slug, name=name, description=description, columns=columns_definition))
        
        dataset._create_columns(dataset_response)
        
        rows_with_ids = dataset._convert_rows_by_names_to_col_ids(df.to_dict(orient="records"))
        
        rows_response = dataset.add_rows(dataset_response.slug, rows_with_ids)
        
        dataset._create_rows(rows_response)
        
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
    
    # API Methods
    
    def create_dataset(self, input: CreateDatasetRequest) -> CreateDatasetResponse:
        """Create new dataset"""
        data = input.model_dump()
        
        result = self._get_http_client().post("projects/default/datasets", data)
        
        if result is None:
            raise Exception("Failed to create dataset")
        
        response = CreateDatasetResponse(**result)

        for field, value in response.model_dump().items():
            if hasattr(self, field) and field != 'columns':
                setattr(self, field, value)

        return response

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

    def add_rows(self, dataset_slug: str, rows: List[ValuesMap]) -> CreateRowsResponse:
        """Add rows to dataset"""
        data = {"rows": rows}
        result = self._get_http_client().post(f"projects/default/datasets/{dataset_slug}/rows", data)
        if result is None:
            raise Exception(f"Failed to add row to dataset {dataset_slug}")
        return CreateRowsResponse(**result)

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