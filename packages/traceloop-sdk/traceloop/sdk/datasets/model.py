import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ColumnType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"


class ColumnDefinition(BaseModel):
    name: str
    type: ColumnType


class CreateDatasetRequest(BaseModel):
    slug: str
    name: Optional[str] = None
    description: Optional[str] = None
    columns: Optional[List[ColumnDefinition]] = None


class CreateDatasetResponse(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    columns: Dict[str, ColumnDefinition]
    last_version: Optional[str] = Field(default=None, alias="lastVersion")
    created_at: datetime.datetime = Field(default=None, alias="createdAt")
    updated_at: datetime.datetime = Field(default=None, alias="updatedAt")


ValuesMap = Dict[str, Any]


class UpdateDatasetInput(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class CreateColumnInput(BaseModel):
    name: str
    type: ColumnType


class UpdateColumnInput(BaseModel):
    name: str
    type: ColumnType


class CreateRowsInput(BaseModel):
    rows: List[ValuesMap]


class RowObject(BaseModel):
    id: str
    rowIndex: float = Field(alias="rowIndex")
    values: ValuesMap
    created_at: datetime.datetime = Field(alias="created_at")
    updated_at: datetime.datetime = Field(alias="updated_at")


class CreateRowsResponse(BaseModel):
    rows: List[RowObject]
    total: int


class UpdateRowInput(BaseModel):
    values: ValuesMap


class DatasetMetadata(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    last_version: Optional[str] = Field(default=None, alias="lastVersion")
    columns: Optional[List[ColumnDefinition]] = None
    created_at: Optional[datetime.datetime]
    updated_at: Optional[datetime.datetime]


class DatasetFullData(BaseModel):
    """Full dataset response matching Go BasicDataset + rows"""
    id: str
    slug: str
    name: Optional[str] = None
    description: Optional[str] = None
    columns: Dict[str, ColumnDefinition]
    rows: List[RowObject]
    created_at: datetime.datetime
    updated_at: datetime.datetime
