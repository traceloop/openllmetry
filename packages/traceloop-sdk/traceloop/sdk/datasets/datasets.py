import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


from traceloop.sdk.client.http import HTTPClient
from traceloop.sdk.datasets.attachment import Attachment, ExternalAttachment
from traceloop.sdk.datasets.dataset import Dataset
from traceloop.sdk.datasets.model import (
    ColumnDefinition,
    ColumnType,
    CreateDatasetRequest,
    CreateDatasetResponse,
    DatasetMetadata,
    ValuesMap,
)

logger = logging.getLogger(__name__)


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

    def create(self, dataset_request: CreateDatasetRequest) -> Dataset:
        """
        Create a dataset with support for initial attachments.

        If row values contain Attachment or ExternalAttachment objects,
        they will be automatically uploaded/attached after dataset creation.

        Args:
            dataset_request: Dataset creation request, can contain Attachment objects in row values

        Returns:
            Created dataset with all attachments processed

        Example:
            dataset_request = CreateDatasetRequest(
                slug="products",
                name="Product Catalog",
                columns=[
                    ColumnDefinition(slug="name", name="Name", type=ColumnType.STRING),
                    ColumnDefinition(slug="image", name="Image", type=ColumnType.FILE),
                ],
                rows=[{
                    "name": "Product A",
                    "image": Attachment(file_path="/path/to/image.jpg", file_type=FileCellType.IMAGE)
                }]
            )
            dataset = datasets.create(dataset_request)
        """
        # Extract attachment objects from rows
        attachments_to_process = self._extract_attachments(dataset_request)

        # Replace attachment objects with None for initial creation
        clean_request = self._prepare_request_for_creation(dataset_request)

        # Create the dataset
        response = self._create_dataset(clean_request)
        dataset = Dataset.from_create_dataset_response(response, self._http)

        # Process attachments if any
        if attachments_to_process:
            self._process_attachments(dataset, attachments_to_process)

        return dataset

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

            # TODO: Handle None case for fieldnames more gracefully
            if reader.fieldnames is None:
                raise ValueError("CSV file has no headers")

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

        # TODO: Pandas returns Hashable keys, should ensure they're strings
        rows = [
            {self._slugify(str(k)): v for k, v in row.items()}
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
        return cast(str, result)

    def get_version_jsonl(self, slug: str, version: str) -> str:
        """Get a specific version of a dataset as a JSONL string"""
        result = self._http.get(f"datasets/{slug}/versions/{version}/jsonl")
        if result is None:
            raise Exception(f"Failed to get dataset {slug} by version {version}")
        return cast(str, result)

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

    def _extract_attachments(
        self, request: CreateDatasetRequest
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract attachment objects from row values.

        Returns:
            Dictionary mapping row index to column slug to attachment object
        """
        attachments: Dict[int, Dict[str, Any]] = {}
        if request.rows:
            for row_idx, row in enumerate(request.rows):
                for col_slug, value in row.items():
                    if isinstance(value, (Attachment, ExternalAttachment)):
                        if row_idx not in attachments:
                            attachments[row_idx] = {}
                        attachments[row_idx][col_slug] = value
        return attachments

    def _prepare_request_for_creation(
        self, request: CreateDatasetRequest
    ) -> CreateDatasetRequest:
        """
        Replace attachment objects with None in row values for initial dataset creation.

        Args:
            request: Original dataset request with potential attachment objects

        Returns:
            Modified request with attachment objects replaced by None
        """
        if not request.rows:
            return request

        # Create a deep copy of rows to avoid modifying the original
        clean_rows = []
        for row in request.rows:
            clean_row: Dict[str, Any] = {}
            for col_slug, value in row.items():
                if isinstance(value, (Attachment, ExternalAttachment)):
                    clean_row[col_slug] = None
                else:
                    clean_row[col_slug] = value
            clean_rows.append(clean_row)

        # Create a new request with cleaned rows
        return CreateDatasetRequest(
            slug=request.slug,
            name=request.name,
            description=request.description,
            columns=request.columns,
            rows=clean_rows,
        )

    def _process_attachments(
        self, dataset: Dataset, attachments: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Upload/attach all attachment objects to their respective cells.

        Args:
            dataset: The created dataset
            attachments: Dictionary mapping row index to column slug to attachment object
        """
        if not dataset.rows:
            return

        for row_idx, row_attachments in attachments.items():
            if row_idx >= len(dataset.rows):
                logger.warning(
                    f"Warning: Row index {row_idx} out of range, skipping attachments"
                )
                continue

            row = dataset.rows[row_idx]
            for col_slug, attachment in row_attachments.items():
                try:
                    if isinstance(attachment, Attachment):
                        ref = attachment.upload(
                            self._http, dataset.slug, row.id, col_slug
                        )
                    elif isinstance(attachment, ExternalAttachment):
                        ref = attachment.attach(
                            self._http, dataset.slug, row.id, col_slug
                        )
                    else:
                        continue

                    # Update row values locally
                    row.values[col_slug] = {
                        "type": ref.file_type.value if ref.file_type else "file",
                        "status": "success",
                        "storage": ref.storage_type.value,
                        "storage_key": getattr(ref, "storage_key", None),
                        "url": getattr(ref, "url", None),
                        "metadata": ref.metadata,
                    }
                except Exception as e:
                    logger.warning(
                        f"Warning: Failed to process attachment for row {row_idx}, column {col_slug}: {e}"
                    )
                    # Mark as failed in row values
                    row.values[col_slug] = {
                        "type": "file",
                        "status": "failed",
                        "error": str(e),
                    }
