from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from app.chains.init import load_llm
from langchain.prompts import (
    ChatPromptTemplate,
)

from app.chains.history import get_session_history


def build(chain_config: {}):
    template = chain_config.get("template")
    if template is None:
        template = """
        You are a highly capable and versatile AI assistantcurate, helpful, and thoughtful responses to any query or request you receive. Follow these guidelines in your interactions:
    
        1. Understand the Query:
           - Carefully analyze the user's input to grasp the core of their question or request.
           - If the query is ambiguous, ask for clarification before proceeding., designed to help users with a wide range of tasks and questions. Your goal is to provide ac
        
        2. Provide Comprehensive Answers:
           - Offer detailed and informative responses, covering all relevant aspects of the question.
           - Break down complex topics into easily understandable parts.
           - Use examples, analogies, or comparisons when appropriate to enhance understanding.
        
        3. Be Objective and Balanced:
           - Present information from multiple perspectives when dealing with complex or controversial topics.
           - Clearly distinguish between facts, opinions, and speculation in your responses.
        
        4. Tailor Your Language:
           - Adjust your language and explanation level based on the context and the user's apparent knowledge level.
           - Use technical terms when appropriate, but always provide explanations for specialized vocabulary.
        
        5. Problem-Solving Approach:
           - For problem-solving tasks, outline a step-by-step approach.
           - Explain the reasoning behind each step or suggestion you provide.
        
        6. Creativity and Brainstorming:
           - When asked for ideas or creative solutions, provide a range of options.
           - Encourage innovative thinking by suggesting unconventional approaches when appropriate.
        
        7. Numerical and Data Analysis:
           - For questions involving numbers or data, provide clear calculations and explain your methodology.
           - Use appropriate units and provide context for numerical information.
        
        8. Coding and Technical Assistance:
           - When helping with coding or technical issues, provide code snippets or pseudo-code where relevant.
           - Explain the logic behind the code and suggest best practices.
        
        9. Research and Citations:
           - If you reference specific facts or studies, mention that you don't have access to real-time information and suggest the user verify the information.
           - Encourage users to seek authoritative sources for critical information.
        
        10. Ethical Considerations:
            - Refrain from assisting with anything illegal or harmful.
            - Provide ethical alternatives if a request raises moral concerns.
        
        11. Limitations and Honesty:
            - Be upfront about your limitations as an AI.
            - If you're unsure about something, clearly state that and suggest ways the user might find more accurate information.
        
        12. Follow-up and Engagement:
            - After providing an answer, ask if the user needs any clarification or has follow-up questions.
            - Encourage users to provide more context if it would help you give a better answer.
        
        Remember, your primary goal is to be helpful, informative, and to enhance the user's understanding or ability to complete their task. Always strive to provide value in every interaction.
        
        Now, please assist the user with their query or task to the best of your ability.
        """

    # build a prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                template
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ]
    )

    # load llm
    llm_config = {
        "temperature": chain_config["temperature"]
    }
    llm = load_llm(llm_config)

    # build chain
    chain = prompt | llm

    # add history
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="user_input",
        history_messages_key="history",
    )
    return runnable_with_history


def build_for_title(chain_config: {}):
    template = """
    You are an AI assistant tasked with generating a concise and relevant title for a conversation based on the user's initial input. Your goal is to create a title that captures the essence of the conversation's topic or main question.

    Given: The first message or query from a user in a conversation.
    
    Task: Generate a short, descriptive title for the conversation based on this initial input.
    
    Guidelines:
    1. Keep the title brief, ideally 3-7 words.
    2. Focus on the main topic or question presented in the user's input.
    3. Use clear and simple language.
    4. Avoid using personal pronouns or unnecessary articles.
    5. If the input is a question, consider rephrasing it into a statement.
    6. Capitalize the first word and any proper nouns.
    
    Examples:
    User Input: "How do I make a chocolate cake from scratch?"
    Title: "Homemade Chocolate Cake Recipe"
    
    User Input: "What were the major causes of World War II?"
    Title: "World War II Major Causes"
    
    User Input: "Can you explain the theory of relativity in simple terms?"
    Title: "Theory of Relativity Simplified"
    
    Now, generate an appropriate title based on the given user input.
    
    User Input: {user_input}
    
    Title:
    """
    title_prompt = PromptTemplate(
        input_variables=["user_input"],
        template=template
    )

    # load llm
    llm_config = {
        "temperature": chain_config["temperature"]
    }
    llm = load_llm(llm_config)

    chain = title_prompt | llm

    return chain


def build_for_summary(chain_config: {}):
    # Define templates for summarizing text chunks and final summary
    template01 = """
    Summarize the key points of  the following text chunk concisely in 2-6 sentences. 
    {pdf_chunk}
    Focus on the main ideas and most important information.
    """

    template02 = """
    Based on the provided summaries of individual text chunks：
    {summary_chunks} 
    
    create a comprehensive two-part summary of the entire document. Your response should consist of:
    Overall Summary
    
    This summary should:
    
    1. Begin with a concise introduction stating the document's main subject and purpose.
    2. Present the key ideas, arguments, and information in a logical flow.
    3. Highlight overarching themes or concepts that span multiple sections of the document.
    4. Provide context where necessary to enhance understanding.
    5. Conclude with the document's main message or significance.
    
    Aim for approximately 500～700 words for this overall summary.
    
    Key Highlights
    
    Identify and present the most significant points or unique aspects of the document. This section should:
    
    1. List 4-6 key highlights or takeaways from the document.
    2. For each highlight:
       a. Clearly state the point in a concise sentence or phrase.
       b. Briefly explain why this point is significant or how it contributes to the document's overall value.
       c. If applicable, provide a brief supporting example from the original text or in-text evidence, noting that the number of the chunk is not cited.
    
    3. Ensure that these highlights represent diverse aspects of the document, such as:
       - Novel ideas or innovative approaches
       - Critical findings or conclusions
       - Unique methodologies or frameworks
       - Surprising or counterintuitive information
       - Practical applications or implications
    
    Present each highlight as a separate bullet point for clarity. Aim for approximately 30-50 words per highlight.
    
    General Guidelines:
    - Maintain objectivity and accurately reflect the original document's tone and perspective.
    - Use clear, accessible language while preserving any essential technical terms.
    - Ensure that the overall summary and highlights complement each other without excessive repetition.
    
    Your final output should provide readers with both a comprehensive understanding of the document's content and a quick reference to its most notable aspects.
    """

    # Create prompt templates with message placeholders
    prompt01 = ChatPromptTemplate.from_messages(
        [
            ("system", template01),
            ("human", "pdf_chunk"),
        ]
    )

    prompt02 = ChatPromptTemplate.from_messages(
        [
            ("system", template02),
            MessagesPlaceholder(variable_name="history"),
            ("human", "summary_chunks"),
        ]
    )

    # Extract LLM configuration and handle potential KeyError
    temperature = chain_config.get("temperature", 0.7)  # Provide a default temperature if not set
    llm_config = {"temperature": temperature}
    llm = load_llm(llm_config)

    # Create chains with history management
    chain01 = prompt01 | llm
    chain02 = prompt02 | llm
    runnable_with_history = RunnableWithMessageHistory(
        chain02,
        get_session_history,
        input_messages_key="summary_chunks",
        history_messages_key="history",
    )

    return [chain01, runnable_with_history]
