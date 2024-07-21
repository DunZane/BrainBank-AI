from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Optional, Sequence
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_qdrant import Qdrant
from langchain_core.embeddings import Embeddings
from qdrant_client.http import models

DEFAULT_COLLECTION_NAME = "langchain_message_store"


class QdrantChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, *,
                 url: str,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 embedding: Embeddings,
                 metadata: str,
                 k: int = 4,
                 ) -> None:
        self.qdrant = Qdrant.from_existing_collection(
            embedding=embedding,
            collection_name=collection_name,
            url=url, )

        self.metadata = metadata
        self.k = k

    @property
    def messages(self) -> List[BaseMessage]:
        """"""
        filter_query = None
        if self.metadata:
            filter_query = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchValue(value=self.metadata.get("file_id"))
                    ),
                    models.FieldCondition(
                        key="metadata.user_id",
                        match=models.MatchValue(value=self.metadata.get("user_id"))
                    )
                ]
            )



        pass

    def clear(self) -> None:
        pass
