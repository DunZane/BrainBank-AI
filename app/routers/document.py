import io
import fitz
from langchain_core.documents import Document

from app import logger
from minio import Minio
from fastapi import APIRouter
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config.document import file_settings, split_settings, qdrant_settings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from app.models.documention import DocumentEmbeddingRequest, DocumentEmbeddingResponse

router = APIRouter(
    prefix="/document",
    tags=["document"]
)


@router.post("/embedding")
async def embedding_file(request: DocumentEmbeddingRequest) -> DocumentEmbeddingResponse:
    logger.info(f"Get http request:{request.dict()}")
    response = DocumentEmbeddingResponse()

    # Create minio client
    minio_client = Minio(
        endpoint=file_settings.ENDPOINT,
        access_key=file_settings.ACCESS_KEY,
        secret_key=file_settings.SECRET_KEY,
        secure=False,
    )

    # Get the object from minio server
    try:
        minio_response = minio_client.get_object(bucket_name=request.bucket_name,
                                                 object_name=request.object_key)
        data_stream = io.BytesIO(minio_response.read())
        logger.info("Load file data from minio server success.")
    except Exception as e:
        logger.error(f"Unexpected error occurred when load file data from minio server: {e}")
        response.is_success = False
        return response

    # Use PyMuPDF to process PDF files
    page_text = ""
    try:
        pdf_file = fitz.open(stream=data_stream, filetype="pdf")
        for page_index in range(len(pdf_file)):
            page = pdf_file[page_index]
            page_text = page_text + page.get_text()
        logger.info("Success read PDF content from minio_response object.")
    except Exception as e:
        logger.error(f"Unexpected error occurred when extract text from PDF: {e}")
        response.is_success = False
        return response

    # Split document into chunks and embedding them into Qdrant
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=split_settings.CHUNK_SIZE,
            chunk_overlap=split_settings.CHUNK_OVERLAP
        )
        metadata = {"user_id": request.user_id}
        document_chunks = text_splitter.create_documents(
            [page_text],
            metadatas=[metadata] * len([page_text])
        )
        Qdrant.from_documents(
            documents=document_chunks,
            embedding=HuggingFaceBgeEmbeddings(),
            base_url=f'http://{qdrant_settings.HOST}:{qdrant_settings.PORT}',
            collection_name=qdrant_settings.COLLECTION_NAME,
        )
        logger.info("Convert data into vector and store in Qdrant success.")
    except Exception as e:
        logger.error(f"Unexpected error occurred when embedding document:{e}")
        response.is_success = False
    finally:
        minio_response.close()
        minio_response.release_conn()

    return response
