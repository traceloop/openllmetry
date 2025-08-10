import csv
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from pydantic import Field, PrivateAttr
import pandas as pd


from traceloop.sdk.dataset.model import (
    ColumnDefinition,
    ValuesMap,
    CreateDatasetRequest,
    CreateDatasetResponse,
    ColumnType,
    DatasetMetadata,
    DatasetFullData,
    DatasetBaseModel
)
from traceloop.sdk.dataset.dataset import Dataset
from traceloop.sdk.client.http import HTTPClient

class Datasets(DatasetBaseModel):
    """
    Datasets class dataset API communication
    """
    _http: HTTPClient


    def __init__(self, http: HTTPClient):
        self._http = http

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

        return Dataset.from_full_data(validated_data, cls._http)


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

        dataset_response = cls._create_dataset(
            CreateDatasetRequest(
                slug=slug,
                name=name,
                description=description,
                columns=columns_definition
            )
        )

        return Dataset.from_create_dataset_response(dataset_response, rows_with_names, cls._http)


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


        dataset_response = cls._create_dataset(
            CreateDatasetRequest(
                slug=slug,
                name=name,
                description=description,
                columns=columns_definition
            )
        )

        rows = df.to_dict(orient="records")

        return Dataset.from_create_dataset_response(dataset_response, rows, cls._http)

    @classmethod
    def get_version_csv(cls, slug: str, version: str) -> str:
        """Get a specific version of a dataset as a CSV string"""
        result = cls._get_http_client_static().get(f"projects/default/datasets/{slug}/versions/{version}")
        if result is None:
            raise Exception(f"Failed to get dataset {slug} by version {version}")
        return result
    
    def _create_dataset(cls, input: CreateDatasetRequest) -> CreateDatasetResponse:
        """Create new dataset"""
        data = input.model_dump()

        result = cls._get_http_client_static().post("projects/default/datasets", data)

        if result is None:
            raise Exception("Failed to create dataset")

        response = CreateDatasetResponse(**result)

        for field, value in response.model_dump().items():
            if hasattr(cls, field) and field != 'columns':
                setattr(cls, field, value)

        return response
