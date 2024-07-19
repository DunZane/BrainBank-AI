from pydantic import BaseModel
from typing import List

from app.model import Message


class ChatRequest(BaseModel):
    messages: List[Message]
    rag: bool = False
    stream: bool = False
    session_id: str


class BotRequest(BaseModel):
    messages: List[Message]
    rag: bool = False
    stream: bool = False
    session_id: str

