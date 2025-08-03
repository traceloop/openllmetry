from enum import Enum
from pydantic import BaseModel, ConfigDict


class ColumnType(str, Enum):
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    NUMBER = "NUMBER"
    JSON = "JSON"


class DatasetBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)