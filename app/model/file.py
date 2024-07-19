from pydantic import BaseModel
from typing import List

from app.model import Message


class FileRequest(BaseModel):
    messages: List[Message]
    stream: bool = False
    session_id: str
    file_id: str
    user_id: str


class FileSummaryRequest(BaseModel):
    stream: bool = False
    session_id: str
    file_id: str
    user_id: str

