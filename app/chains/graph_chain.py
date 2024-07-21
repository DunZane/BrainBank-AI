from typing import Dict, Any

from app.chains.init import load_llm, load_neo4j_graph, load_embedding
from app.chains.history import get_session_history

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough, RunnableParallel

from app.chains.retriever.neo4j import Neo4jRetriever


def build(chain_config: Dict[str, Any], metadata=Dict[str, Any]):
    # load embedding
    embedding, _ = load_embedding()
    # Initialize LLM with the given configuration
    llm_config = {"temperature": chain_config.get("temperature", 0.7)}  # Set a default value if not provided
    llm = load_llm(llm_config)

    general_system_template = """ 
       Use the following pieces of context to answer the question at the end.
       The context contains question-answer pairs and their links from Stackoverflow.
       You should prefer information from accepted or more upvoted answers.
       Make sure to rely on information from the answers and not on questions to provide accurate responses.
       When you find particular answer in the context useful, make sure to cite it in the answer using the link.
       If you don't know the answer, just say that you don't know, don't try to make up an answer.
       ----
       {summaries}
       ----
       Each answer you generate should contain a section at the end of links to 
       Stackoverflow questions and answers you found useful, which are described under Source value.
       You can only use links to StackOverflow questions that are present in the context and always
       add links to the end of the answer in the style of citations.
       Generate concise answers with references sources section of links to 
       relevant StackOverflow questions only at the end of the answer.
       """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    retriever = Neo4jRetriever()

    chain = (
            RunnableParallel({"summaries": retriever,
                              "question": RunnablePassthrough(),
                              "history": RunnablePassthrough()})
            | qa_prompt
            | llm
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history


def build_in_ticket(chain_config: {}):
    neo4j_graph = load_neo4j_graph()
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3")

    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. \n{question[0]}\n----\n\n"
        questions_prompt += f"{question[1][:150]}\n\n"
        questions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in formulating high quality questions. 
    Formulate a question in the same style and tone as the following example questions.
    {questions_prompt}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return format template:
    ---
    Title: This is a new title
    Question: This is a new question
    ---
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following template format or you will be unplugged.
                ---
                Title: New title
                Question: New question
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{user_input}"),
        ]
    )

    # Initialize LLM with the given configuration
    llm_config = {"temperature": chain_config.get("temperature", 0.7)}  # Set a default value if not provided
    llm = load_llm(llm_config)

    chain = chat_prompt | llm

    return chain
