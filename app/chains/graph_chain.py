from langchain_core.runnables import RunnableWithMessageHistory

from app.chains.init import load_llm, load_embedding, load_kg, load_neo4j_graph

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import RetrievalQAWithSourcesChain

from app.routers import get_session_history


def build(chain_config: {}):
    # Initialize LLM with the given configuration
    llm_config = {"temperature": chain_config.get("temperature", 0.7)}  # Set a default value if not provided
    llm = load_llm(llm_config)

    embeddings, _ = load_embedding()
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

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    query = ('\n'
             '    WITH node AS question, score AS similarity\n'
             '       CALL  { with question\n'
             '           MATCH (question)<-[:ANSWERS]-(answer)\n'
             '           WITH answer\n'
             '           ORDER BY answer.is_accepted DESC, answer.score DESC\n'
             '           WITH collect(answer)[..2] as answers\n'
             '           RETURN reduce(str=\'\', answer IN answers | str + \n'
             '                   \'\n### Answer (Accepted: \'+ answer.is_accepted +\n'
             '                   \' Score: \' + answer.score+ \'): \'+  answer.body + \'\n\') as answerTexts\n'
             '       } \n'
             '       RETURN \'##Question: \' + question.title + \'\n\' + question.body + \'\n\' \n'
             '           + answerTexts AS text, similarity as score, {source: question.link} AS metadata\n'
             '       ORDER BY similarity ASC // so that best answers are the last\n'
             '    ')
    kg = load_kg(embeddings, query)

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
        memory=get_session_history
    )
    return kg_qa


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
