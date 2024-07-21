import os
from langchain_community.chat_message_histories import ElasticsearchChatMessageHistory

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


def get_es_session_history(session_id: str) -> BaseChatMessageHistory:
    ES_BASE_URL = os.getenv("ES_BASE_URL")
    ES_PWD = os.getenv("ES_PWD")
    CHAT_HISTORY_INDEX = os.getenv("CHAT_HISTORY_INDEX")

    return ElasticsearchChatMessageHistory(
        es_url=ES_BASE_URL,
        es_password=ES_PWD,
        index=CHAT_HISTORY_INDEX,
        session_id=session_id
    )
