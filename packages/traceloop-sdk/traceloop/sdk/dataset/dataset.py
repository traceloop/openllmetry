import csv
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import Field, PrivateAttr
import pandas as pd
import os

from traceloop.sdk.dataset.model import (
    ColumnDefinition,
    ValuesMap,
    CreateDatasetRequest,
    CreateDatasetResponse,
    CreateRowsResponse,
    ColumnType,
    DatasetMetadata,
    RowObject,
    DatasetFullData,
    PublishDatasetResponse,
    AddColumnResponse,
    DatasetBaseModel
)
from .column import Column
from .row import Row
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__


class Dataset(DatasetBaseModel):
    """
    Dataset class dataset API communication
    """
    id: Optional[str] = Field(default=None, alias="id")
    name: Optional[str] = None
    slug: str
    description: Optional[str] = None
    columns: Optional[List[Column]] = Field(default_factory=list)
    rows: Optional[List[Row]] = Field(default_factory=list)
    last_version: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    _http: HTTPClient

    def __init__(self, http: HTTPClient):
        self._http = http
    
    @classmethod
    def from_full_data(cls, full_data: DatasetFullData, http: HTTPClient) -> "Dataset":
        """Create a Dataset instance from DatasetFullData"""
        dataset_data = full_data.model_dump(exclude={'columns', 'rows'})
        dataset = cls(http=http)
        
        for field, value in dataset_data.items():
            if hasattr(dataset, field):
                setattr(dataset, field, value)
        
        dataset._create_columns(full_data.columns)
        dataset._create_rows(full_data.rows)
        
        return dataset
    
    @classmethod
    def from_create_dataset_response(cls, response: CreateDatasetResponse, rows: List[ValuesMap], http: HTTPClient) -> "Dataset":
        """Create a Dataset instance from CreateDatasetResponse"""
        dataset = cls(http=http)

        dataset._create_columns(response.columns)

        rows_with_ids = dataset._convert_rows_by_names_to_col_ids(rows)

        dataset.add_rows_api(rows_with_ids)

        return dataset

    def _convert_rows_by_names_to_col_ids(
        self, rows_with_names: List[ValuesMap]
    ) -> List[ValuesMap]:
        """Convert multiple rows from column names to column IDs"""
        # Create a mapping from column names to column IDs
        name_to_id = {col.name: col.id for col in self.columns}
        return [
            {name_to_id[col_name]: val for col_name, val in row_data.items() if col_name in name_to_id}
            for row_data in rows_with_names
        ]

    def publish(self) -> str:
        """Publish dataset"""
        result = self._http.post(
            f"projects/default/datasets/{self.slug}/publish", {}
        )
        if result is None:
            raise Exception(f"Failed to publish dataset {self.slug}")
        return PublishDatasetResponse(**result).version

    def add_column(self, name: str, col_type: ColumnType) -> Column:
        """Add new column (returns Column object)"""
        data = {
            "name": name,
            "type": col_type
        }

        result = self._http.post(
            f"projects/default/datasets/{self.slug}/columns",
            data
        )
        if result is None:
            raise Exception(f"Failed to add column to dataset {self.slug}")
        col_response = AddColumnResponse(**result)
    
        column = Column(
            _http=self._http,
            id=col_response.id,
            name=col_response.name,
            type=col_response.type,
            dataset_id=self.id
        )
        self.columns.append(column)
        return column

    def _create_columns(self, raw_columns: Dict[str, ColumnDefinition]):
        """Create Column objects from API response which includes column IDs"""
        for column_id, column_def in raw_columns.items():
            column = Column(
                _http=self._http,
                id=column_id,
                name=column_def.name,
                type=column_def.type,
                dataset_id=self.id
            )
            column._client = self
            self.columns.append(column)

    def _create_rows(self, raw_rows: List[RowObject]):
        for _, row_obj in enumerate(raw_rows):
            row = Row(
                id=row_obj.id,
                values=row_obj.values,
                dataset_id=self.id
            )
            row._client = self
            self.rows.append(row)
       

    def update_column_api(
        self, column_id: str,
        data: Dict[str, str]
    ) -> Dict[str, Any]:
        """Update column properties"""
        result = self._http.put(
            f"projects/default/datasets/{self.slug}/columns/{column_id}",
            data
        )
        if result is None:
            raise Exception(f"Failed to update column {column_id}")
        return result

    # Row APIs

    def add_rows_api(self, rows: List[ValuesMap]) -> CreateRowsResponse:
        """Add rows to dataset"""
        data = {"rows": rows}
        result = self._http.post(
            f"projects/default/datasets/{self.slug}/rows",
            data
        )
        if result is None:
            raise Exception(f"Failed to add row to dataset {self.slug}")

        response = CreateRowsResponse(**result)
        self._create_rows(response.rows)
        return response

    def delete_row_api(self, row_id: str) -> None:
        """Delete row"""
        result = self._http.delete(f"projects/default/datasets/{self.slug}/rows/{row_id}")
        if result is None:
            raise Exception(f"Failed to delete row {row_id}")

    def update_row_api(self, row_id: str, values: Dict[str, Any]):
        """Update row values"""
        data = {"values": values}
        self._http.put(
            f"projects/default/datasets/{self.slug}/rows/{row_id}",
            data
        )
