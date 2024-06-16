from langchain_core.chat_history import BaseChatMessageHistory
from langchain_elasticsearch.chat_history import ElasticsearchChatMessageHistory

from app.config.application import settings


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return ElasticsearchChatMessageHistory(
        es_url=settings.ES_URL,
        es_password=settings.ES_PWD,
        index=settings.CHAT_HISTORY_INDEX,
        session_id=session_id
    )