import csv
from typing import List, Optional
from pathlib import Path

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


from traceloop.sdk.dataset.model import (
    ColumnDefinition,
    ValuesMap,
    CreateDatasetRequest,
    CreateDatasetResponse,
    ColumnType,
    DatasetMetadata,
    DatasetFullData,
)
from traceloop.sdk.dataset.dataset import Dataset
from traceloop.sdk.client.http import HTTPClient


class Datasets:
    """
    Datasets class dataset API communication
    """

    _http: HTTPClient

    def __init__(self, http: HTTPClient):
        self._http = http

    def get_all(self) -> List[DatasetMetadata]:
        """List all datasets metadata"""
        result = self._http.get("datasets")
        if result is None:
            raise Exception("Failed to get datasets")
        return [DatasetMetadata(**dataset) for dataset in result]

    def delete_by_slug(self, slug: str) -> None:
        """Delete dataset by slug without requiring an instance"""
        success = self._http.delete(f"datasets/{slug}")
        if not success:
            raise Exception(f"Failed to delete dataset {slug}")

    def get_by_slug(self, slug: str) -> "Dataset":
        """Get a dataset by slug and return a full Dataset instance"""
        result = self._http.get(f"datasets/{slug}")
        if result is None:
            raise Exception(f"Failed to get dataset {slug}")

        validated_data = DatasetFullData(**result)

        return Dataset.from_full_data(validated_data, self._http)

    def from_csv(
        self,
        file_path: str,
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "Dataset":
        """Create dataset from CSV file"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        columns_definition: List[ColumnDefinition] = []
        rows_with_names: List[ValuesMap] = []

        with open(file_path, "r", encoding="utf-8") as csvfile:
            # Detect delimiter
            sample = csvfile.read(1024)
            csvfile.seek(0)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter

            reader = csv.DictReader(csvfile, delimiter=delimiter)

            for field_name in reader.fieldnames:
                columns_definition.append(
                    ColumnDefinition(
                        name=field_name,
                        type=ColumnType.STRING,
                    )
                )

            for _, row_data in enumerate(reader):
                rows_with_names.append(dict(row_data))

        dataset_response = self._create_dataset(
            CreateDatasetRequest(
                slug=slug,
                name=name,
                description=description,
                columns=columns_definition,
            )
        )

        dataset = Dataset.from_create_dataset_response(
            dataset_response, rows_with_names, self._http
        )
        return dataset

    def from_dataframe(
        self,
        df: "pd.DataFrame",
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "Dataset":
        """Create dataset from pandas DataFrame"""
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

            columns_definition.append(ColumnDefinition(name=col_name, type=col_type))

        dataset_response = self._create_dataset(
            CreateDatasetRequest(
                slug=slug,
                name=name,
                description=description,
                columns=columns_definition,
            )
        )

        rows = df.to_dict(orient="records")

        return Dataset.from_create_dataset_response(dataset_response, rows, self._http)

    def get_version_csv(self, slug: str, version: str) -> str:
        """Get a specific version of a dataset as a CSV string"""
        result = self._http.get(f"datasets/{slug}/versions/{version}")
        if result is None:
            raise Exception(f"Failed to get dataset {slug} by version {version}")
        return result

    def _create_dataset(self, input: CreateDatasetRequest) -> CreateDatasetResponse:
        """Create new dataset"""
        data = input.model_dump()

        result = self._http.post("datasets", data)

        if result is None:
            raise Exception("Failed to create dataset")

        return CreateDatasetResponse(**result)
