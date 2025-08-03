from typing import Any, Literal, Optional

import bleach
from pydantic import BaseModel, Field, validator

from rag import SearchQueryType


class Usage(BaseModel):
    """Token usage information."""

    total_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        description="Total tokens used in the request",
    )
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        description="Tokens used for the prompt",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        le=1000000000,
        description="Tokens used for the completion",
    )


class SourceMetadata(BaseModel):
    """Metadata associated with a document source."""

    language: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Language of the document",
    )
    date_created: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Creation date of the document",
    )
    last_modified: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Last modification date",
    )
    page_number: int = Field(
        0,
        ge=-1,
        le=1000000,
        description="Page number in the document",
    )
    description: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Description of the document content",
    )
    height: int = Field(
        0,
        ge=0,
        le=100000,
        description="Height of the document in pixels",
    )
    width: int = Field(
        0,
        ge=0,
        le=100000,
        description="Width of the document in pixels",
    )
    location: list[float] = Field(
        default=[], description="Bounding box location of the content"
    )
    location_max_dimensions: list[int] = Field(
        default=[], description="Maximum dimensions of the document"
    )
    content_metadata: dict[str, Any] = Field(
        default={}, description="Metadata about the content"
    )


class SourceResult(BaseModel):
    """Represents a single source document result."""

    document_id: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Unique identifier of the document",
    )
    content: str = Field(
        default="",
        pattern=r"[\s\S]*",
        description="Extracted content from the document",
    )
    document_name: str = Field(
        default="",
        max_length=100000,
        pattern=r"[\s\S]*",
        description="Name of the document",
    )
    document_type: Literal["image", "text", "table", "chart", "audio"] = Field(
        default="text", description="Type of document content"
    )
    score: float = Field(default=0.0, description="Relevance score of the document")

    metadata: SourceMetadata


class Citations(BaseModel):
    """Represents the sources section of the API response."""

    total_results: int = Field(
        default=0,
        ge=0,
        le=1000000,
        description="Total number of source documents found",
    )
    results: list[SourceResult] = Field(
        default=[], description="List of document results"
    )


class Message(BaseModel):
    """Definition of the Chat Message type."""

    role: Literal["user", "assistant", "system", None] = Field(
        description="Role for a message: either 'user' or 'assistant' or 'system",
        default="user",
    )
    content: str = Field(
        description="The input query/prompt to the pipeline.",
        default="Hello! What can you help me with?",
        max_length=131072,
        pattern=r"[\s\S]*",
    )

    @validator("role")
    @classmethod
    def validate_role(cls, value):
        """Field validator function to validate values of the field role"""
        if value:
            value = bleach.clean(value, strip=True)
            valid_roles = {"user", "assistant", "system"}
            if value is not None and value.lower() not in valid_roles:
                raise ValueError("Role must be one of 'user', 'assistant', or 'system'")
            return value.lower()

    @validator("content")
    @classmethod
    def sanitize_content(cls, v):
        """Feild validator function to santize user populated feilds from HTML"""
        return bleach.clean(v, strip=True)


class ChainResponseChoices(BaseModel):
    """Definition of Chain response choices"""

    index: int = Field(default=0, ge=0, le=256)
    message: Message = Field(default=Message(role="assistant", content=""))
    delta: Message = Field(default=Message(role=None, content=""))
    finish_reason: Optional[str] = Field(
        default=None, max_length=4096, pattern=r"[\s\S]*"
    )


class ChainResponse(BaseModel):
    """Definition of Chain APIs resopnse data type"""

    id: str = Field(default="", max_length=100000, pattern=r"[\s\S]*")
    choices: list[ChainResponseChoices] = Field(default_factory=list)
    # context will be deprecated once `sources` field is implemented and populated
    model: str = Field(default="", max_length=4096, pattern=r"[\s\S]*")
    object: str = Field(default="", max_length=4096, pattern=r"[\s\S]*")
    created: int = Field(default=0, ge=0, le=9999999999)
    # Place holder fields for now to match generate API response structure
    usage: Optional[Usage] = Field(
        default=Usage(), description="Token usage statistics"
    )
    citations: Optional[Citations] = Field(
        default=Citations(), description="Source documents used for the response"
    )


class Prompt(BaseModel):
    messages: list[Message] = Field(min_length=1)
    use_knowledge_base: bool = Field(default=True)
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    max_tokens: int = Field(default=1024, ge=0, le=128000)
    query_type: SearchQueryType = Field(default="auto")


class Ingest(BaseModel):
    texts: list[str] | None = None
    chunk_size: int = 200
    chunk_overlap: int = 0
