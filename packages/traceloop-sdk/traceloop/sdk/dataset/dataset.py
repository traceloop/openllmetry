from typing import List, Optional, Dict
from pydantic import Field

from traceloop.sdk.dataset.model import (
    ColumnDefinition,
    ValuesMap,
    CreateDatasetResponse,
    CreateRowsResponse,
    ColumnType,
    RowObject,
    PublishDatasetResponse,
    AddColumnResponse,
)
from .column import Column
from .row import Row
from .base import BaseDatasetEntity
from traceloop.sdk.client.http import HTTPClient


class Dataset(BaseDatasetEntity):
    """
    Dataset class dataset API communication
    """

    id: str
    name: str
    slug: str
    description: str
    columns: List[Column] = Field(default_factory=list)
    rows: Optional[List[Row]] = Field(default_factory=list)
    last_version: Optional[str] = None

    def __init__(self, http: HTTPClient):
        super().__init__(http)
        self.columns = []
        self.rows = []

    @classmethod
    def from_create_dataset_response(
        cls, response: CreateDatasetResponse, http: HTTPClient
    ) -> "Dataset":
        """Create a Dataset instance from CreateDatasetResponse"""
        dataset = cls(http=http)
        for field, value in response.model_dump(exclude={"columns", "rows"}).items():
            setattr(dataset, field, value)

        dataset._create_columns(response.columns)

        if response.rows:
            dataset._create_rows(response.rows)

        return dataset

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

    def add_column(self, slug: str, name: str, col_type: ColumnType) -> Column:
        """Add new column (returns Column object)"""
        data = {"slug": slug, "name": name, "type": col_type}

        result = self._http.post(f"datasets/{self.slug}/columns", data)
        if result is None:
            raise Exception(f"Failed to add column to dataset {self.slug}")
        col_response = AddColumnResponse(**result)

        column = Column(
            http=self._http,
            dataset=self,
            slug=col_response.slug,
            name=col_response.name,
            type=col_response.type,
        )
        self.columns.append(column)
        return column

    def _create_columns(self, raw_columns: Dict[str, ColumnDefinition]):
        """Create Column objects from API response which includes column IDs"""
        for column_slug, column_def in raw_columns.items():
            column = Column(
                http=self._http,
                dataset=self,
                slug=column_slug,
                name=column_def.name,
                type=column_def.type,
            )
            self.columns.append(column)

    def _create_rows(self, raw_rows: List[RowObject]):
        for _, row_obj in enumerate(raw_rows):
            row = Row(
                http=self._http,
                dataset=self,
                id=row_obj.id,
                values=row_obj.values,
            )
            if self.rows:
                self.rows.append(row)
            else:
                self.rows = [row]
