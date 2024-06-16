from pydantic import BaseModel


class DocumentEmbeddingRequest(BaseModel):
    user_id: str = ""
    bucket_name: str = "file"
    object_key: str = ""


class DocumentEmbeddingResponse(BaseModel):
    is_success: bool = True
