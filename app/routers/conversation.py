from fastapi import APIRouter
from app.utils.token import token_usage
from app.utils.history import get_session_history
from app.llm.openai import title_build_llm_chain, message_build_llm_chain
from app.models.conversation import MessageRequest, ConversationTitleRequest
from app.views.chat_view import ConversationTitleResponse, MessageResponse
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from app import logger

router = APIRouter(
    prefix="/conversation",
    tags=["conversation"],
)


@router.post("/title", description="Chat with LLM using a specified template to generate a title.")
async def chat_with_llm_title(request: ConversationTitleRequest) -> ConversationTitleResponse:
    try:
        logger.info(f"Received request:{request.dict()}")

        # TODO: Unified Template Management
        template = """
                Based on the user posed question: {question} to generate only one title.
                But there are a few requirements:
                - The language of the title needs to match the language of the input.
                - The generated title cannot duplicate the elements in the following list：{excluded_titles}
                - Titles need be concise and clear.
                - Give the title directly and don't give any other information.
                """

        logger.info("Creating prompt template...")
        prompt = PromptTemplate(
            input_variables=["question", "excluded_titles"],
            template=template,
        )

        logger.info("Building LLM chain...")
        chain = title_build_llm_chain(prompt)

        inputs = {"question": request.content, "excluded_titles": request.excluded_titles}

        logger.info("Invoking LLM chain...")
        resp = chain.invoke(input=inputs)

        logger.info(f"Get response from LLM chain: {resp}")
    except Exception as e:
        resp = ""
        logger.error(f"Error occur:{str(e)}")

    msg = ConversationTitleResponse(content=resp)
    return msg


@router.post("/message", description="Generate a response message based on user input.")
async def chat_with_llm(request: MessageRequest) -> MessageResponse:
    logger.info(f"Creating response message, receiving request: {request.dict()}")

    if request.conversation_mode == "":
        request.conversation_mode = "primary_assistant"
    logger.info(f"Conversation mode is: {request.conversation_mode}")

    chunks = []
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an AI assistant.",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        logger.info(f"Prompt is structured successfully: {prompt}")

        chain = message_build_llm_chain(prompt)

        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        logger.info("Build history llm chain successfully.")

        inputs = {"character": request.conversation_mode, "input": request.content,
                  "session_id": request.conversation_id}

        async for chunk in with_message_history.astream(input=inputs,
                                                        config={
                                                            "configurable": {"session_id": request.conversation_id}}):
            chunks.append(chunk)
        logger.info(f"Get the response from LLM: {chunks}")
    except Exception as e:
        logger.error(e)

    resp = MessageResponse(content=chunks, token_usage=token_usage(chunks))
    return resp


