import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_astradb import AstraDBChatMessageHistory


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
    ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    ASTRA_DB_NAME_SPACE = os.getenv("ASTRA_DB_NAME_SPACE")

    return AstraDBChatMessageHistory(
        session_id=session_id,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=ASTRA_DB_NAME_SPACE,
    )
