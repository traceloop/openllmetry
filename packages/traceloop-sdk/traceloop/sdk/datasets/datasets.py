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
        if isinstance(result, dict) and "datasets" in result:
            return [DatasetMetadata(**dataset) for dataset in result["datasets"]]
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

        validated_data = CreateDatasetResponse(**result)

        return Dataset.from_create_dataset_response(validated_data, self._http)

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
                        slug=self._slugify(field_name),
                        name=field_name,
                        type=ColumnType.STRING,
                    )
                )

            for _, row_data in enumerate(reader):
                rows_with_names.append(
                    {self._slugify(k): v for k, v in row_data.items()}
                )

        dataset_response = self._create_dataset(
            CreateDatasetRequest(
                slug=slug,
                name=name,
                description=description,
                columns=columns_definition,
                rows=rows_with_names,
            )
        )

        dataset = Dataset.from_create_dataset_response(dataset_response, self._http)
        return dataset

    def from_dataframe(
        self,
        df: "pd.DataFrame",
        slug: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "Dataset":
        """Create dataset from pandas DataFrame"""
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for from_dataframe. Install with: pip install pandas"
            )

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

            columns_definition.append(
                ColumnDefinition(
                    slug=self._slugify(col_name), name=col_name, type=col_type
                )
            )

        rows = [
            {self._slugify(k): v for k, v in row.items()}
            for row in df.to_dict(orient="records")
        ]

        dataset_response = self._create_dataset(
            CreateDatasetRequest(
                slug=slug,
                name=name,
                description=description,
                columns=columns_definition,
                rows=rows,
            )
        )

        return Dataset.from_create_dataset_response(dataset_response, self._http)

    def get_version_csv(self, slug: str, version: str) -> str:
        """Get a specific version of a dataset as a CSV string"""
        result = self._http.get(f"datasets/{slug}/versions/{version}")
        if result is None:
            raise Exception(f"Failed to get dataset {slug} by version {version}")
        return result

    def get_version_jsonl(self, slug: str, version: str) -> str:
        """Get a specific version of a dataset as a JSONL string"""
        result = self._http.get(f"datasets/{slug}/versions/{version}/jsonl")
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

    def _slugify(self, name: str) -> str:
        """Slugify a name"""
        import re

        if not name:
            raise ValueError("Name cannot be empty")

        slug = name.lower()

        # Replace spaces and underscores with hyphens
        slug = slug.replace(" ", "-").replace("_", "-")

        # Remove any character that's not alphanumeric or hyphen
        slug = re.sub(r"[^a-z0-9-]+", "", slug)

        # Remove multiple consecutive hyphens
        slug = re.sub(r"-+", "-", slug)

        # Trim hyphens from start and end
        slug = slug.strip("-")

        if not slug:
            raise ValueError(f"Name '{name}' cannot be slugified to a valid slug")

        return slug
