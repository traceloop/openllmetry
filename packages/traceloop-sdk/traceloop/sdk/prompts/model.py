import datetime
from typing import List, Optional

from pydantic import BaseModel


class TemplateEngine:
    JINJA2 = "jinja2"


class RegistryObjectBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class Message(RegistryObjectBaseModel):
    index: int
    role: str
    template: str
    variables: Optional[List[str]] = []


class ModelConfig(RegistryObjectBaseModel):
    mode: str
    model: str
    temperature: float
    max_tokens: Optional[int]
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
    llm_config: ModelConfig


class Target(RegistryObjectBaseModel):
    id: str
    updated_at: datetime.datetime
    prompt_id: str
    version: str


class Prompt(RegistryObjectBaseModel):
    id: str
    versions: List[PromptVersion]
    target: Target
    key: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
