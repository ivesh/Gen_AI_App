import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#LANGSMITH TRACKING
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_API_KEY']="true"
os.environ['LANGCHAIN_PROJECT']=os.getenv("LANGCHAIN_PROJECT")

#Designing prompt template to tell LLM what to do
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

## Streamlit framework:
st.title("Chatbot with Langchian Expression Language(LCEL)")
input_text=st.text_input("What question do you have in mind?")

# Sidebar for additional info
st.sidebar.markdown("### How to Use:")
st.sidebar.markdown("""
    Chatbot Features: This chatbot is an example of how different components of langchain is used. \n
    Component 1: Chat Prompt Template: Used to give instructions/values to LLM for processing the output. \n
    Component 2: LLM to be used: Various paid LLM and opensource LLM can be used depending on the project. \n
    Component 3: Output parser: To display the result in a structured format. \n
    Final Step: Take the user question pass it into ChatPromptTemplate, activate the LLM and pass it to the output parser. 
                    """)

## Calling ollama laama2 model, more info-> https://github.com/ollama/ollama
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt | llm | output_parser

## If input_text and enter then invoke the chain with question 
if input_text:
    st.write(chain.invoke({"question":input_text})) 