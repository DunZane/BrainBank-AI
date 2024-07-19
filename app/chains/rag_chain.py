from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableWithMessageHistory, RunnableParallel, RunnablePassthrough

from qdrant_client.http import models
from qdrant_client import models

from app.chains.init import load_llm, load_qdrant_client
from app.routers import get_session_history
from app import logger


class QdrantRetriever(BaseRetriever):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        logger.info(query)

        # 确保 query 是字符串
        if not isinstance(query, str):
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

        qdrant = load_qdrant_client()
        found_docs = qdrant.similarity_search_with_score(query, filter=filter_query,k=2)
        if found_docs:
            pdf_content = " ".join(doc.page_content for doc, _ in found_docs)
        else:
            pdf_content = ""
        return [Document(page_content=pdf_content)]


def build(chain_config: dict, metadata: dict):
    # Template moved to a separate constant for clarity
    template = """
    You are a professional AI assistant specialized in answering questions about a specific PDF document. 
    You have carefully read and understood the entire content of this document. 
    Please answer the user's questions according to the following guidelines:

    1. Answer questions based solely on the information contained in the PDF document. If a question goes beyond the scope of the document, politely inform the user.
    2. If the answer appears directly in the document, cite the relevant paragraph. Use quotation marks to indicate directly quoted text.
    3. If multiple pieces of information from the document need to be synthesized to answer a question, clearly explain your reasoning process.
    4. If the information in the document is insufficient to fully answer a question, state this and provide relevant information that is available in the document.
    5. If the user's question is unclear, politely ask for clarification.
    6. Keep answers concise and clear, but provide sufficient detail to comprehensively answer the question.
    7. If the document contains charts or images, mention these visual elements when answering related questions.
    If the user inquires about your capabilities or knowledge source, explain that you are an AI assistant answering questions based on a specific PDF document.

    Remember, your goal is to provide accurate, helpful answers based solely on the given PDF document:{pdf_content}
    
    {user_input}
    
    """

    # Initialize LLM with the given configuration
    llm_config = {"temperature": chain_config.get("temperature", 0.7)}  # Set a default value if not provided
    llm = load_llm(llm_config)

    # Create chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        # MessagesPlaceholder(variable_name="history"),
        ("system","history"),
        ("human", "{user_input}"),
    ])

    qdrant_retriever = QdrantRetriever(metadata)

    # build chain
    chain = (
            RunnableParallel({"pdf_content": qdrant_retriever,
                              "user_input": RunnablePassthrough(),
                              "history": RunnablePassthrough()})
            | prompt
            | llm
    )

    # Add memory to chain using RunnableWithMessageHistory
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="user_input",
        history_messages_key="history",
    )

    return runnable_with_history
