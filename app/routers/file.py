import json
from fastapi import APIRouter
from queue import Queue, Empty
from threading import Thread
from collections.abc import Generator
from starlette.responses import StreamingResponse

from app import logger
from app.chains import rag_chain
from app.model.file import FileRequest, FileSummaryRequest

router = APIRouter(
    prefix="/v1",
    tags=["file"],
)


@router.post("/pdf-bot")
async def pdf_bot(request: FileRequest):
    logger.info(f"Received request: {request.messages}")
    message_result = {
        "system_content": None,
        "user_content": None
    }

    for message in request.messages:
        if message.role == "system":
            message_result["system_content"] = message.content
        elif message.role == "user":
            message_result["user_content"] = message.content
    chain_config = {"template": None, "temperature": 0}
    if message_result["system_content"] != "":
        chain_config["template"] = message_result["system_content"]

    metadata = {"file_id": request.file_id, "user_id": request.user_id}

    chain_with_history = rag_chain.build(chain_config=chain_config, metadata=metadata)

    content = message_result["user_content"]
    if request.stream:
        return stream_response(chain_with_history, content, request.session_id)
        # return list_response(chain_with_history, content, request.session_id)
    else:
        return regular_response(chain_with_history, content, request.session_id)


@router.post("/pdf-bot")
async def pdf_summary(request: FileSummaryRequest):
    pass


def regular_response(chain, content: str, session_id: str):
    """
    Handles regular (non-streaming) chat responses.
    """
    # Prepare input for the chain
    inputs = {"user_input": content}

    output = chain.invoke(
        input=inputs,
        config={"configurable": {"session_id": session_id}}
    )

    # Return the response as a JSONResponse
    return {"content": output.content}


def stream_response(chain, content: str, session_id: str):
    """
    Handles streaming chat responses using Server-Sent Events (SSE).
    """

    def task():
        try:
            inputs = {"user_input": content}
            for token in chain.stream(input=inputs, config={"configurable": {"session_id": session_id}}):
                logger.info(f"Token content: {token.content}")
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


def list_response(chain, content: str, session_id: str):
    inputs = {"user_input": content}

    list_token = []
    for token in chain.stream(
            input=inputs,
            config={"configurable": {"session_id": session_id}}
    ):
        list_token.append(token.content)

    return {"content": list_token}
