import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class RegistryObjectBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Message(RegistryObjectBaseModel):
    index: int
    role: str
    template: str
    variables: Optional[List[str]] = []


class ModelConfig(RegistryObjectBaseModel):
    mode: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    stop: List[str]
    frequency_penalty: float
    presence_penalty: float


class PromptVersion(RegistryObjectBaseModel):
    id: str
    hash: str
    version: int
    name: Optional[str] = None
    created_at: datetime.datetime
    provider: str
    templating_engine: str
    messages: List[Message]
    model_config: ModelConfig


class Prompt(RegistryObjectBaseModel):
    id: str
    versions: List[PromptVersion]
    key: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
