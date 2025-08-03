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
    columns: Optional[List[ColumnDefinition]] = None
    created_at: datetime
    updated_at: datetime


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


class UpdateRowInput(BaseModel):
    values: ValuesMap


