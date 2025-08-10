from datetime import datetime
from typing import List, Optional, Dict

from traceloop.sdk.dataset.model import (
    ColumnDefinition,
    ValuesMap,
    CreateDatasetResponse,
    CreateRowsResponse,
    ColumnType,
    RowObject,
    DatasetFullData,
    PublishDatasetResponse,
    AddColumnResponse,
)
from .column import Column
from .row import Row
from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.version import __version__


class Dataset:
    """
    Dataset class dataset API communication
    """

    id: str
    name: str
    slug: str
    description: str
    columns: List[Column] = None
    rows: Optional[List[Row]] = None
    last_version: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    _http: HTTPClient

    def __init__(self, http: HTTPClient):
        self._http = http
        self.columns = []
        self.rows = []

    @classmethod
    def from_full_data(cls, full_data: DatasetFullData, http: HTTPClient) -> "Dataset":
        """Create a Dataset instance from DatasetFullData"""
        dataset_data = full_data.model_dump(exclude={"columns", "rows"})
        dataset = cls(http=http)

        # Set all attributes from the dataset data
        for field, value in dataset_data.items():
            setattr(dataset, field, value)

        dataset._create_columns(full_data.columns)
        if full_data.rows:
            dataset._create_rows(full_data.rows)

        return dataset

    @classmethod
    def from_create_dataset_response(
        cls, response: CreateDatasetResponse, rows: List[ValuesMap], http: HTTPClient
    ) -> "Dataset":
        """Create a Dataset instance from CreateDatasetResponse"""
        dataset = cls(http=http)
        for field, value in response.model_dump(exclude={"columns"}).items():
            setattr(dataset, field, value)

        dataset._create_columns(response.columns)

        rows_with_ids = dataset._convert_rows_by_names_to_col_ids(rows)

        dataset.add_rows(rows_with_ids)

        return dataset

    def _convert_rows_by_names_to_col_ids(
        self, rows_with_names: List[ValuesMap]
    ) -> List[ValuesMap]:
        """Convert multiple rows from column names to column IDs"""
        # Create a mapping from column names to column IDs
        name_to_id = {col.name: col.id for col in self.columns}
        return [
            {
                name_to_id[col_name]: val
                for col_name, val in row_data.items()
                if col_name in name_to_id
            }
            for row_data in rows_with_names
        ]

    def publish(self) -> str:
        """Publish dataset"""
        result = self._http.post(f"datasets/{self.slug}/publish", {})
        if result is None:
            raise Exception(f"Failed to publish dataset {self.slug}")
        return PublishDatasetResponse(**result).version

    def add_rows(self, rows: List[ValuesMap]) -> None:
        """Add rows to dataset"""
        result = self._http.post(f"datasets/{self.slug}/rows", {"rows": rows})
        if result is None:
            raise Exception(f"Failed to add row to dataset {self.slug}")

        response = CreateRowsResponse(**result)
        self._create_rows(response.rows)

    def add_column(self, name: str, col_type: ColumnType) -> Column:
        """Add new column (returns Column object)"""
        data = {"name": name, "type": col_type}

        result = self._http.post(f"datasets/{self.slug}/columns", data)
        if result is None:
            raise Exception(f"Failed to add column to dataset {self.slug}")
        col_response = AddColumnResponse(**result)

        column = Column(
            http=self._http,
            dataset=self,
            id=col_response.id,
            name=col_response.name,
            type=col_response.type,
            dataset_id=self.id,
        )
        self.columns.append(column)
        return column

    def _create_columns(self, raw_columns: Dict[str, ColumnDefinition]):
        """Create Column objects from API response which includes column IDs"""
        for column_id, column_def in raw_columns.items():
            column = Column(
                http=self._http,
                dataset=self,
                id=column_id,
                name=column_def.name,
                type=column_def.type,
                dataset_id=self.id,
            )
            self.columns.append(column)

    def _create_rows(self, raw_rows: List[RowObject]):
        for _, row_obj in enumerate(raw_rows):
            row = Row(
                http=self._http,
                dataset=self,
                id=row_obj.id,
                values=row_obj.values,
                dataset_id=self.id,
            )
            self.rows.append(row)
