from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    role: str
    content: str


class BaseRequest(BaseModel):
    messages: List[Message]


class BaseResponse(BaseModel):
    pass


class Question(BaseModel):
    pass
