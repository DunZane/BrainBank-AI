from fastapi import APIRouter, Depends

from app.chains import graph_chain
from app.model.ticket import BaseTicket

from app import logger

router = APIRouter(
    prefix="/v1",
    tags=["ticket"],
)


@router.post("/generate-ticket")
async def generate_ticket(request: BaseTicket):
    chain = graph_chain.build_in_ticket(chain_config={})

    inputs = {"user_input":request.text}

    llm_response = chain.invoke(inputs).content
    logger.info(llm_response)

    lines = llm_response.strip().split("\n")

    title = ""
    question = ""
    is_question = False  # flag to know if we are inside a "Question" block

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = (
                True  # set the flag to True once we encounter a "Question:" line
            )
        elif is_question:
            # if the line does not start with "Question:" but we are inside a "Question" block,
            # then it is a continuation of the question
            question += "\n" + line.strip()

    return {"content": {"title": title, "question": question}}
