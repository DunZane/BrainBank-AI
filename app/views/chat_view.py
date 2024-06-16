from pydantic import BaseModel
from typing import List


class ConversationTitleResponse(BaseModel):
    content: str


class MessageResponse(BaseModel):
    content: List[str]
    token_usage: int
