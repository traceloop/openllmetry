from enum import Enum
from pydantic import BaseModel
from typing import List, Optional


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


