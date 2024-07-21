import os
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_qdrant import Qdrant as qdrant
from langchain_qdrant.vectorstores import Qdrant
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from qdrant_client import models
from langchain_core.documents import Document

from app.chains.init import load_embedding
from app import logger


class QdrantRetriever(BaseRetriever):
    qdrant_client: Optional[Qdrant] = None
    embeddings: Optional[Embeddings] = None
    metadata: dict = {}
    QDRANT_BASE_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "content_embedding"

    def __init__(self, metadata: Optional[dict] = None):
        super().__init__()
        self.embeddings, _ = load_embedding()
        self.metadata = metadata or {}
        self.QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", self.QDRANT_BASE_URL)
        self.QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", self.QDRANT_COLLECTION_NAME)
        self.qdrant_client = self._initialize_qdrant_client()

    def _initialize_qdrant_client(self) -> Qdrant:
        try:
            qdrant_client = qdrant.from_existing_collection(
                embedding=self.embeddings,
                collection_name=self.QDRANT_COLLECTION_NAME,
                url=self.QDRANT_BASE_URL
            )
            logger.info("Qdrant client initialized successfully.")
            return qdrant_client
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        # Ensure query is a string
        query = str(query)

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

        try:
            found_docs = self.qdrant_client.similarity_search_with_score(query, filter=filter_query, k=2)
            logger.info(f"Found {len(found_docs)} documents matching the query.")
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return [Document(page_content="")]

        if found_docs:
            pdf_content = " ".join(doc.page_content for doc, _ in found_docs)
        else:
            pdf_content = ""
        return [Document(page_content=pdf_content)]
