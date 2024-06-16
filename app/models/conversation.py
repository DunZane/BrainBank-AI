from pydantic import BaseModel


class ConversationTitleRequest(BaseModel):
    content: str
    excluded_titles: str = ""


class MessageRequest(BaseModel):
    content: str
    conversation_mode: str
    conversation_id: str
