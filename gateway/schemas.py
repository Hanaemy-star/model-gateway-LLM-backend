from pydantic import BaseModel, Field
from typing import Literal

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str  = Field(min_length=1)

class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = Field(default=False)
