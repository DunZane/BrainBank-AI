import io
import json
import os
from typing import List, Dict, Optional, Any
from queue import Queue, Empty
from threading import Thread
import PyPDF2
from starlette.responses import StreamingResponse
from collections.abc import Generator
from langchain.chains.base import Chain
from minio import Minio
from minio.error import S3Error
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


class PDFSummarizationPipeline:
    def __init__(self, chains: List[Chain], metadata: Optional[Dict[str, str]] = None,
                 chunk_size: int = 4096, chunk_overlap: int = 32,
                 minio_client: Optional[Minio] = None):
        self.chains = chains
        self.metadata = metadata or {}
        self.MINIO_BASE_URL = os.getenv("MINIO_BASE_URL", "http://localhost:9000")
        self.MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "default")
        self.MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "default")
        self.MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "default")
        self.minio_client = minio_client or self._initialize_minio_client()
        self.TEXT_SPLITTER = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def _initialize_minio_client(self) -> Minio:
        try:
            minio_client = Minio(
                endpoint=self.MINIO_BASE_URL,
                access_key=self.MINIO_ACCESS_KEY,
                secret_key=self.MINIO_SECRET_KEY,
                secure=self.MINIO_BASE_URL.startswith("https")
            )
            logger.info("Minio client initialized successfully.")
            return minio_client
        except S3Error as e:
            logger.error(f"Failed to initialize Minio client: {e}")
            raise

    @staticmethod
    def _process_pdf(file_data: bytes) -> str:
        try:
            pdf_stream = io.BytesIO(file_data)
            reader = PyPDF2.PdfReader(pdf_stream)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
            return text
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return ""

    def _fetch_pdf_from_minio(self) -> Optional[str]:
        if not self.minio_client.bucket_exists(self.MINIO_BUCKET_NAME):
            logger.warning(f"Bucket {self.MINIO_BUCKET_NAME} does not exist.")
            return None

        try:
            response = self.minio_client.get_object(self.MINIO_BUCKET_NAME, self.metadata.get("file_name"))
            file_data = response.read()
            response.close()
            response.release_conn()
            return self._process_pdf(file_data)
        except (S3Error, Exception) as e:
            logger.error(f"Error fetching file from Minio: {e}")
            return None

    def _preprocess(self) -> Dict[str, List[str]]:
        text = self._fetch_pdf_from_minio()
        if not text:
            raise ValueError("Invalid text content, please check the file object in Minio")

        chunks = self.TEXT_SPLITTER.split_text(text=text)
        logger.info(f"Number of chunks: {len(chunks)}")
        return {"chunks": chunks}

    def _forward(self, model_inputs: Dict[str, List[str]]) -> Dict:
        chunks = model_inputs.get("chunks", [])
        summary_chunks = {}

        for i, chunk in enumerate(chunks):
            inputs = {"pdf_chunk": chunk}
            try:
                output = self.chains[0].invoke(input=inputs)
                summary_chunks[f"chunk-{i}"] = output.content  # Use json data format input model
                logger.info(f"Processed chunk with output: {output.content}")
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")

        return summary_chunks

    def _postprocess(self, summary_chunks: Dict, stream: bool) -> Any:
        json_str = json.dumps(summary_chunks, indent=4)
        inputs = {"summary_chunks": json_str}

        try:
            if stream:
                def task():
                    try:
                        for token in self.chains[1].stream(input=inputs, config=
                        {"configurable": {"session_id": self.metadata.get("session_id")}}):
                            q.put(token.content)
                    except Exception as e:
                        logger.error(f"Error in task: {e}")
                    finally:
                        q.put(None)  # Signal that the task is done

                q = Queue()
                Thread(target=task).start()

                def generate() -> Generator:
                    accumulated_content = ""
                    while True:
                        try:
                            next_token = q.get(timeout=120)
                            if next_token is None:
                                break
                            accumulated_content += next_token
                            yield f"data: {json.dumps({'content': accumulated_content})}\n\n"
                        except Empty:
                            yield f"data: {json.dumps({'content': accumulated_content})}\n\n"

                return StreamingResponse(generate(), media_type="text/event-stream")
            else:
                response = self.chains[1].invoke(input=inputs, config={
                    "configurable": {"session_id": self.metadata.get("session_id")}})
                result = response  # Ensure this is the expected result
                logger.info(f"Processed summary with output: {result}")
                return {"content": result.content}
        except Exception as e:
            logger.error(f"Error in postprocessing: {e}")
            return None

    def process(self, stream: bool = False) -> Any:
        """
        Process the PDF file and return the summarized content.
        """
        preprocessed = self._preprocess()
        model_outputs = self._forward(preprocessed)
        return self._postprocess(model_outputs, stream)

    def __call__(self, stream: bool = False, **kwargs) -> Any:
        """
        Call the process method to perform the summarization pipeline.
        """
        return self.process(stream=stream)
