import os

from app import logger
from internal.chatglm import Chatglm6b

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import BedrockChat

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import Neo4jVector
from langchain_qdrant import Qdrant

from langchain_community.graphs import Neo4jGraph


def load_llm(llm_config: dict):
    # load params from envs
    llm_model = os.getenv("LLM")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")

    # load model param
    temperature = llm_config.get("temperature")

    # return llm by params
    if llm_model == 'gpt-4':
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=temperature, model_name="gpt-4", streaming=True)
    elif llm_model == 'gpt-3.5':
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_model == "claudev2":
        logger.info("LLM: ClaudeV2")
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": temperature, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    elif llm_model == "chatglm-6b":
        logger.info("LLM: Chatglm-6b")
        return Chatglm6b(
            temperature=temperature,
            base_url="http://localhost:5001",
            streaming=True,
            top_k=10,
            top_p=0.3,
            max_length=3072,
        )
    else:
        logger.info(f"LLM: Using Ollama: {llm_model}")
        return ChatOllama(
            temperature=temperature,
            base_url=ollama_base_url,
            model=llm_model,
            streaming=True,
            top_k=10,
            top_p=0.3,
            num_ctx=3072,
        )


def load_embedding():
    # load params
    embedding_model = os.getenv("EMBEDDING_MODEL")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")

    # return embedding by params
    if embedding_model == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=ollama_base_url, model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")

    elif embedding_model == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    elif embedding_model == "google-genai-embedding-001":
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        dimension = 768
        logger.info("Embedding: Using Google Generative AI Embeddings")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="/Users/zhaodeng/.cache/huggingface/hub"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_kg(embeddings, query):
    neo4jVector = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database="neo4j",  # neo4j by default
        index_name="stackoverflow",  # vector by default
        text_node_property="body",  # text by default
        retrieval_query=query,
    )
    return neo4jVector


def load_neo4j_graph():
    # load params from env
    url = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")

    # create a client
    neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
    _, dimension = load_embedding()

    # some init check
    index_query = "CALL db.index.vector.createNodeIndex('stackoverflow', 'Question', 'embedding', $dimension, 'cosine')"
    try:
        neo4j_graph.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass
    index_query = "CALL db.index.vector.createNodeIndex('top_answers', 'Answer', 'embedding', $dimension, 'cosine')"
    try:
        neo4j_graph.query(index_query, {"dimension": dimension})
    except:  # Already exists
        pass

    return neo4j_graph


def load_qdrant_client():
    # load params from env
    QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL")
    QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")

    # load embedding
    embeddings, _ = load_embedding()

    qdrant = Qdrant.from_existing_collection(
        embedding=embeddings,
        collection_name=QDRANT_COLLECTION_NAME,
        url=QDRANT_BASE_URL,)

    return qdrant
