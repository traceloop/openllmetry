import datetime
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class TemplateEngine:
    JINJA2 = "jinja2"


class RegistryObjectBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class TextContent(RegistryObjectBaseModel):
    type: Literal["text"]
    text: str


class Url(RegistryObjectBaseModel):
    url: str


class ImageContent(RegistryObjectBaseModel):
    type: Literal["image_url"]
    image_url: Url


Content = Annotated[Union[TextContent, ImageContent], Field(discriminator="type")]


class Message(RegistryObjectBaseModel):
    index: int
    role: str
    template: Union[str, List[Content]]
    variables: Optional[List[str]] = []


class ToolFunction(RegistryObjectBaseModel):
    name: str
    description: str
    parameters: dict


class Tool(RegistryObjectBaseModel):
    type: Literal["function"]
    function: ToolFunction


class ResponseFormat(RegistryObjectBaseModel):
    type: Literal["json_schema"]
    json_schema: dict


class ModelConfig(RegistryObjectBaseModel):
    mode: str
    model: str
    temperature: float
    max_tokens: Optional[int] = None
    top_p: float
    stop: List[str]
    frequency_penalty: float
    presence_penalty: float
    tool_choice: Optional[Literal["none", "auto", "required"]] = None
    tools: Optional[List[Tool]] = None
    response_format: Optional[ResponseFormat] = None


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
