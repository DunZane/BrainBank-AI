from pydantic_settings import BaseSettings


class FileSettings(BaseSettings):
    # minio settings
    ENDPOINT: str = "127.0.0.1:9000"
    ACCESS_KEY: str = "U30jdGSRn18kXzYRFmQ2"
    SECRET_KEY: str = "NGV8Cn66xwdUul6EyTAdd0QPmdEZYr0ogfCSK06s"



class SplitSettings(BaseSettings):
    # split file settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50


class QdrantSettings(BaseSettings):
    # qdrant settings
    HOST: str = "127.0.0.1"
    PORT: int = "6333"
    COLLECTION_NAME: str = "documents"


file_settings = FileSettings()
split_settings = SplitSettings()
qdrant_settings = QdrantSettings()
