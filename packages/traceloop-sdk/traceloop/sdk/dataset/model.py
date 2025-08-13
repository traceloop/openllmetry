import datetime
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ColumnType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"


class ColumnDefinition(BaseModel):
    slug: Optional[str] = None
    name: str
    type: ColumnType


ValuesMap = Dict[str, Any]


class CreateDatasetRequest(BaseModel):
    slug: str
    name: Optional[str] = None
    description: Optional[str] = None
    columns: Optional[List[ColumnDefinition]] = None
    rows: Optional[List[ValuesMap]] = None


class RowObject(BaseModel):
    id: str
    values: ValuesMap
    created_at: datetime.datetime
    updated_at: datetime.datetime


class CreateDatasetResponse(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    columns: Dict[str, ColumnDefinition]
    rows: Optional[List[RowObject]] = None
    last_version: Optional[str] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime


class UpdateDatasetInput(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class CreateColumnInput(BaseModel):
    slug: str
    name: str
    type: ColumnType


class UpdateColumnInput(BaseModel):
    name: Optional[str] = None
    type: Optional[ColumnType] = None


class CreateRowsInput(BaseModel):
    rows: List[ValuesMap]


class CreateRowsResponse(BaseModel):
    rows: List[RowObject]
    total: int


class PublishDatasetResponse(BaseModel):
    dataset_id: str
    version: str


class AddColumnResponse(BaseModel):
    slug: str
    name: str
    type: ColumnType


class UpdateRowInput(BaseModel):
    values: ValuesMap


class DatasetMetadata(BaseModel):
    id: str
    slug: str
    name: str
    description: Optional[str] = None
    last_version: Optional[str] = None
    columns: Optional[Dict[str, ColumnDefinition]] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
