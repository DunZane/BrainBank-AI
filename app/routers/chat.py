import json
from threading import Thread
from fastapi import APIRouter
from queue import Queue, Empty
from collections.abc import Generator
from starlette.responses import StreamingResponse

from app import logger
from app.model import BaseRequest
from app.model.chat import ChatRequest, BotRequest
from app.chains import llm_chain, graph_chain

router = APIRouter(
    prefix="/v1",
    tags=["chat"],
)


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Handles chat requests. Supports both regular and streaming responses.
    """
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

    chain_with_history = llm_chain.build(chain_config=chain_config)

    content = message_result["user_content"]
    if request.stream:
        return stream_response(chain_with_history, content, request.session_id)
        # return list_response(chain_with_history, content, request.session_id)
    else:
        return regular_response(chain_with_history, content, request.session_id)


@router.get("/title")
async def title(request: BaseRequest):
    """
    Based on input message to create a title requests.
    """
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
    title_chain = llm_chain.build_for_title(chain_config=chain_config)

    message_title = title_chain.invoke(input=message_result["user_content"]).content
    return {"title": message_title}


@router.post("/bot")
async def bot(request: BotRequest):
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

    chain = graph_chain.build(chain_config=chain_config)

    content = message_result["user_content"]
    if request.stream:
        return stream_response(chain,content,request.session_id)
    else:
        return regular_response(chain,content,request.session_id)


def regular_response(chain, content: str, session_id: str):
    """
    Handles regular (non-streaming) chat responses.
    """
    # Prepare input for the chain
    inputs = {"user_input": content}

    # Invoke the chain with the given session ID
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
    q = Queue()

    def task():
        inputs = {"user_input": content}
        for token in chain.stream(
                input=inputs,
                config={"configurable": {"session_id": session_id}}
        ):
            q.put(token.content)
        q.put(None)  # Signal that the task is done

    Thread(target=task).start()

    def generate() -> Generator:
        accumulated_content = ""
        while True:
            try:
                next_token = q.get(timeout=20)  # Increased timeout
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
