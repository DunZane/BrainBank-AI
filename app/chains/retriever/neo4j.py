import os
from typing import List, Optional

from langchain_community.vectorstores import Neo4jVector
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings

from app.chains.init import load_embedding
from app import logger


class Neo4jRetriever(BaseRetriever):
    neo4j_vector: Optional[Neo4jVector] = None
    embedding: Optional[Embeddings] = None
    NEO4J_BASE_URI: str = "http://localhost:7687"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "system"
    NEO4J_INDEX_NAME: str = "default"

    def __init__(self):
        super().__init__()
        self.embedding, _ = load_embedding()
        self.NEO4J_BASE_URI = os.getenv("NEO4J_BASE_URI", self.NEO4J_BASE_URI)
        self.NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", self.NEO4J_USERNAME)
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", self.NEO4J_PASSWORD)
        self.NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", self.NEO4J_DATABASE)
        self.NEO4J_INDEX_NAME = os.getenv("NEO4J_INDEX_NAME", self.NEO4J_INDEX_NAME)

        # Ensure all required environment variables are set
        if not all([self.NEO4J_BASE_URI, self.NEO4J_USERNAME, self.NEO4J_PASSWORD,
                    self.NEO4J_DATABASE, self.NEO4J_INDEX_NAME]):
            raise ValueError("One or more environment variables are not set.")

        self.neo4j_vector = self._load_neo4j_vector()

    def _load_neo4j_vector(self) -> Neo4jVector:
        try:
            neo4j_vector = Neo4jVector.from_existing_index(
                embedding=self.embedding,
                url=self.NEO4J_BASE_URI,
                username=self.NEO4J_USERNAME,
                password=self.NEO4J_PASSWORD,
                database=self.NEO4J_DATABASE,
                index_name=self.NEO4J_INDEX_NAME,
            )
            logger.info("Neo4j vector initialized successfully.")
            return neo4j_vector
        except Exception as e:
            logger.error(f"Error loading Neo4j vector: {e}")
            raise

    @staticmethod
    def results_to_string(results: List[dict]) -> str:
        result_strings = []
        for result in results:
            text = result.get("text", "")
            metadata = result.get("metadata", {})
            metadata_str = ', '.join(f'{key}: {value}' for key, value in metadata.items())
            result_str = f"Text:\n{text}\nMetadata:\n{metadata_str}\n"
            result_strings.append(result_str)
        return "\n---\n".join(result_strings)

    def _get_relevant_documents(self, query: dict, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        logger.info(f"Neo4jRetriever got input: {query}")

        query_vector = self.embedding.embed_query(query.get('user_input'))

        cypher_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector) YIELD node, score
        WHERE node:Question
        MATCH (node)<-[:ANSWERS]-(answer)
        WITH node, answer, score
        ORDER BY answer.is_accepted DESC, answer.score DESC
        WITH node, collect(answer)[..2] as answers, score
        WITH node, reduce(str='', answer IN answers | str +
            '\n### Answer (Accepted: ' + toString(answer.is_accepted) +
            ' Score: ' + toString(answer.score) + '): ' + answer.body + '\n') as answerTexts, score
        RETURN '## Question: ' + node.title + '\n' + node.body + '\n' + answerTexts AS text, score, {source: node.link} AS metadata
        """
        try:
            params = {
                "index_name": self.NEO4J_INDEX_NAME,
                "query_vector": query_vector,
                "top_k": 3,
            }
            results = self.neo4j_vector.query(cypher_query, params=params)
            logger.info(f"Query results: {results}")
        except Exception as e:
            logger.error(f"Error executing retrieval process: {e}")
            results = []  # Ensure results is a list even in case of failure

        content = self.results_to_string(results)
        return [Document(page_content=content)]
