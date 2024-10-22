import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Function to initialize LLM
def initialize_llm():
    """Calling the llm from Groq"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="mixtral-8x7b-32768", groq_api_key=groq_api_key)
    return llm

# Function to define a chat prompt with LLM
def chat_prompt(llm, user_message, actor, chat_history):
    """Prompt for chat"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant answering any questions asked in detail. To perform your task better, initially you may ask questions in the form of fill-in-the-blanks to get maximum value from the user, then act as an experienced {Actor} along with references of why you are saying what you are saying."),
        MessagesPlaceholder(variable_name="messages")
    ])
    chain = prompt | llm
    response = chain.invoke({"messages": chat_history + [HumanMessage(content=user_message)], "Actor": actor})
    return response

# Function to handle chat history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = {}
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Function to trim messages
def trim_message(llm):
    trimmer = trim_messages(
        max_tokens=45,
        strategy="last",
        token_counter=llm,
        include_system=True,
        allow_partial=False,
        start_on="human"
    )
    messages = [
        SystemMessage(content="You're a good assistant"),
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="Nice"),
        HumanMessage(content="What's 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="Thanks"),
        AIMessage(content="No problem!"),
        HumanMessage(content="Having fun?"),
        AIMessage(content="Yes!"),
    ]
    trimmed_messages = trimmer.invoke(messages)
    
    # Creating the chain with trimmed messages
    chain = (
        RunnablePassthrough().assign(messages=trimmed_messages)
        | llm
    )
    response = chain.invoke(
        {
            "messages": trimmed_messages + [HumanMessage(content="What ice cream do I like?")],
            "language": "English"
        }
    )
    return response.content

# Streamlit UI implementation
def main():
    st.title("AI Chatbot with LLM")
    st.markdown("Interact with an AI chatbot that can answer your questions in detail and maintain chat history.")
    
    # User input fields
    actor = st.text_input("Enter the role of the assistant (e.g., Data Scientist, Teacher, etc.):", value="Data Scientist")
    user_message = st.text_input("Enter your message:")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize LLM
    llm = initialize_llm()

    # If user enters a message, process it
    if user_message:
        with st.spinner("Processing..."):
            # Convert chat history to list of HumanMessage and AIMessages for context
            chat_history_messages = [
                HumanMessage(content=msg) if speaker == "You" else AIMessage(content=msg)
                for speaker, msg in st.session_state.chat_history
            ]
            response = chat_prompt(llm, user_message, actor, chat_history_messages)
            st.session_state.chat_history.append(("You", user_message))
            st.session_state.chat_history.append(("AI", response.content))
    
    # Display chat history
    st.write("### Chat History:")
    for speaker, message in st.session_state.chat_history:
        st.write(f"**{speaker}:** {message}")

    # Save chat history in the sidebar
    st.sidebar.write("### Saved Chat History:")
    for speaker, message in st.session_state.chat_history:
        st.sidebar.write(f"**{speaker}:** {message}")

    # Trim message button
    if st.button("Trim Messages"):
        with st.spinner("Trimming messages..."):
            trimmed_response = trim_message(llm)
            st.write("### Trimmed Response:")
            st.write(trimmed_response)

if __name__ == "__main__":
    main()
