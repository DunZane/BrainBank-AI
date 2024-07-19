from pydantic import BaseModel


class BaseTicket(BaseModel):
    text: str
