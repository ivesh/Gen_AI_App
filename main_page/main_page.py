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

#Designing prompt template to telll LLM what to do
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

## Streamlit framework:
st.title("Langchian Demo with LLAMA2")
input_text=st.text_input("What question do you have in mind?")

## Calling ollama laama2 model, more info-> https://github.com/ollama/ollama
llm=Ollama(model="llama2")
output_parser=StrOutputParser()
chain=prompt | llm | output_parser

## If input_text and enter then invoke the chain with question 
if input_text:
    st.write(chain.invoke({"question":input_text})) 