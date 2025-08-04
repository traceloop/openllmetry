from enum import Enum
from pydantic import BaseModel, ConfigDict


class ColumnType(str, Enum):
    STRING = "string"
    BOOLEAN = "boolean"
    NUMBER = "number"
    JSON = "json"


class DatasetBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)
