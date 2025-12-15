import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ColumnType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    JSON = "json"
    FILE = "file"


class FileCellType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"


class FileStorageType(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


class FileCellMetadata(BaseModel):
    file_name: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    uploaded_at: Optional[datetime.datetime] = None
    thumbnail_url: Optional[str] = None
    thumbnail_key: Optional[str] = None


class FileCellValue(BaseModel):
    type: FileCellType
    status: str  # "in_progress", "success", "failed"
    storage: FileStorageType
    storage_key: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[FileCellMetadata] = None


# Internal models for API communication (not exposed directly to users)
class UploadURLRequest(BaseModel):
    type: FileCellType
    file_name: str
    content_type: Optional[str] = None
    with_thumbnail: bool = False
    metadata: Optional[Dict[str, Any]] = None


class UploadURLResponse(BaseModel):
    upload_url: str
    thumbnail_upload_url: Optional[str] = None
    storage_key: str
    thumbnail_key: Optional[str] = None
    expires_at: datetime.datetime
    method: str = "PUT"


class UploadStatusRequest(BaseModel):
    status: str  # "success", "failed"
    metadata: Optional[Dict[str, Any]] = None


class ExternalURLRequest(BaseModel):
    type: FileCellType
    url: str
    metadata: Optional[Dict[str, Any]] = None


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
