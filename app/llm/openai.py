from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import Runnable


def title_build_llm_chain(prompt: PromptTemplate):
    """
    Constructs an LLM chain for generating titles or similar text based on the given prompt template.

    Parameters:
        - prompt: Prompt
            The template used to generate prompts for the language model.
    Returns:
        - LLM Chain object
            A language model chain for generating titles
    """
    llm = ChatOpenAI()

    return prompt | llm | StrOutputParser()


def message_build_llm_chain(prompt: ChatPromptTemplate):
    """
     Constructs an LLM chain with history for generating messages based on the given prompt template.

     Parameters:
    - prompt: ChatPromptTemplate
        The template used to generate prompts for the language model, including conversation history.

    Returns:
    - LLM chain
        A language model chain with history for generating messages or conversational text.
    """
    llm = ChatOpenAI()

    runnable: Runnable = prompt | llm | StrOutputParser()

    return runnable
