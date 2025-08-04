import csv
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import Field, PrivateAttr
import pandas as pd
import os

from traceloop.sdk.datasets.model import (
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
    _http: Optional[HTTPClient] = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._http = self._get_http_client()

    def _get_http_client(self) -> HTTPClient:
        """Get or create HTTP client instance"""
        if self._http is None:
            self._http = self._get_http_client_static()
        return self._http

    def _convert_rows_by_names_to_col_ids(
        self, rows_with_names: List[ValuesMap]
    ) -> List[ValuesMap]:
        """Convert multiple rows from column names to column IDs"""
        return [
            {column.id: val for column, val in zip(self.columns, row_data.values())}
            for row_data in rows_with_names
        ]

    def _create_columns(self, raw_columns: Dict[str, ColumnDefinition]):
        """Create Column objects from API response which includes column IDs"""
        for column_id, column_def in raw_columns.items():
            column = Column(
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

    @classmethod
    def _get_http_client_static(cls) -> HTTPClient:
        """Get HTTP client instance for static operations"""
        api_key = os.environ.get("TRACELOOP_API_KEY", "")
        api_endpoint = os.environ.get("TRACELOOP_BASE_URL", "https://api.traceloop.com")

        if not api_key:
            raise ValueError("TRACELOOP_API_KEY environment variable is required")

        return HTTPClient(
            base_url=api_endpoint,
            api_key=api_key,
            version=__version__
        )

    @classmethod
    def get_all(cls) -> List[DatasetMetadata]:
        """List all datasets metadata"""
        result = cls._get_http_client_static().get("projects/default/datasets")
        if result is None:
            raise Exception("Failed to get datasets")
        return [DatasetMetadata(**dataset) for dataset in result]

    @classmethod
    def delete_by_slug(cls, slug: str) -> None:
        """Delete dataset by slug without requiring an instance"""
        success = cls._get_http_client_static().delete(
            f"projects/default/datasets/{slug}"
        )
        if not success:
            raise Exception(f"Failed to delete dataset {slug}")

    @classmethod
    def get_by_slug(cls, slug: str) -> "Dataset":
        """Get a dataset by slug and return a full Dataset instance"""
        result = cls._get_http_client_static().get(f"projects/default/datasets/{slug}")
        if result is None:
            raise Exception(f"Failed to get dataset {slug}")

        validated_data = DatasetFullData(**result)

        dataset_data = validated_data.model_dump(exclude={'columns', 'rows'})
        dataset = Dataset(**dataset_data)

        dataset._create_columns(validated_data.columns)
        dataset._create_rows(validated_data.rows)

        return dataset

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

            for field_name in reader.fieldnames:
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
        )

        dataset_response = dataset.create_dataset(
            CreateDatasetRequest(
                slug=slug,
                name=name,
                description=description,
                columns=columns_definition
            )
        )

        dataset._create_columns(dataset_response.columns)

        rows_with_ids = dataset._convert_rows_by_names_to_col_ids(rows_with_names)

        dataset.add_rows_api(rows_with_ids)

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
        )

        dataset_response = dataset.create_dataset(
            CreateDatasetRequest(
                slug=slug,
                name=name,
                description=description,
                columns=columns_definition
            )
        )

        dataset._create_columns(dataset_response.columns)

        rows_with_ids = dataset._convert_rows_by_names_to_col_ids(
            df.to_dict(orient="records")
        )

        dataset.add_rows_api(rows_with_ids)

        return dataset

    @classmethod
    def get_version_csv(cls, slug: str, version: str) -> str:
        """Get a specific version of a dataset as a CSV string"""
        result = cls._get_http_client_static().get(f"projects/default/datasets/{slug}/versions/{version}")
        if result is None:
            raise Exception(f"Failed to get dataset {slug} by version {version}")
        return result

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
        col_response = self.add_column_api(name, col_type)

        column = Column(
            id=col_response.id,
            name=col_response.name,
            type=col_response.type,
            dataset_id=self.id
        )
        column._client = self
        self.columns.append(column)
        return column

    def create_dataset(self, input: CreateDatasetRequest) -> CreateDatasetResponse:
        """Create new dataset"""
        data = input.model_dump()

        result = self._http.post("projects/default/datasets", data)

        if result is None:
            raise Exception("Failed to create dataset")

        response = CreateDatasetResponse(**result)

        for field, value in response.model_dump().items():
            if hasattr(self, field) and field != 'columns':
                setattr(self, field, value)

        return response

    # Column APIs

    def add_column_api(self, name: str, col_type: str) -> AddColumnResponse:
        """Add column to dataset"""
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
        return AddColumnResponse(**result)

    def delete_column_api(self, column_id: str) -> None:
        """Delete column"""
        result = self._http.delete(
            f"projects/default/datasets/{self.slug}/columns/{column_id}"
        )
        if result is None:
            raise Exception(f"Failed to delete column {column_id}")

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
        result = self._get_http_client().delete(f"projects/default/datasets/{self.slug}/rows/{row_id}")
        if result is None:
            raise Exception(f"Failed to delete row {row_id}")

    def update_row_api(self, row_id: str, values: Dict[str, Any]):
        """Update row values"""
        data = {"values": values}
        self._get_http_client().put(
            f"projects/default/datasets/{self.slug}/rows/{row_id}",
            data
        )
