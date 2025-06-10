import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# Initialize global variables for embeddings and LLM
def initialize_embeddings():
    """Initialize the HuggingFace embeddings model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def initialize_llm():
    """Initialize the LLM using the Ollama model."""
    return Ollama(model="Qwen/Qwen2.5-Omni-7B")

# Load the document from the web
def load_and_split_document(url):
    """Load the web document and split it into smaller chunks."""
    loader = WebBaseLoader(url)
    web_docs = loader.load()
    
    # Split the document into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    web_docs_chunks = text_splitter.split_documents(web_docs)
    return web_docs_chunks

# Create the FAISS vector store
def create_vector_store(doc_chunks, embeddings):
    """Create a FAISS vector store from document chunks."""
    return FAISS.from_documents(doc_chunks, embeddings)

# Create document chain with LLM and prompt
def create_document_chain(llm):
    """Create a document chain to process documents and provide answers."""
    prompt = ChatPromptTemplate.from_template(
        """ 
        Answer the following question based on the provided context:
        <context>
        {context}
        </context>
        """
    )
    return create_stuff_documents_chain(llm, prompt)

# Main retrieval chain logic
def create_and_run_retrieval_chain(query, vector_store, llm):
    """Create and run the retrieval chain for the user query."""
    # Retrieve relevant documents
    retriever = vector_store.as_retriever()
    document_chain = create_document_chain(llm)
    
    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Run the retrieval chain and get the response
    response = retrieval_chain.invoke({"input": query})
    return response

# Streamlit UI setup
def main():
    st.set_page_config(page_title="Web Document Q&A", layout="wide")

    st.title("Interactive Web Document Q&A")
    st.markdown("Use this app to ask questions about any website link that contains text. It converts it into documents internally, and gets AI-generated accurate answers on the topic!")
    
    # Input URL for document loading
    url_input = st.text_input("Enter the URL of the website you want to load for Q&A:")
    query_input = st.text_input("Enter your question on the topic present in the website:")

    # Sidebar for additional info
    st.sidebar.markdown("### How to Use:")
    st.sidebar.markdown("""
    1. Enter a website URL in the input field.
    2. Enter a question about the content of the website.
    3. Get an AI-generated answer based on the document content.
    """)
    
    # Initialize the LLM and embeddings
    embeddings = initialize_embeddings()
    llm = initialize_llm()

    # If URL is entered, load and process the document
    if url_input and query_input:
        with st.spinner("Loading and processing the document..."):
            doc_chunks = load_and_split_document(url_input)
            vector_store = create_vector_store(doc_chunks, embeddings)
        
        # Generate response using the retrieval chain
        with st.spinner("Generating answer..."):
            response = create_and_run_retrieval_chain(query_input, vector_store, llm)
            st.write("### Response:")
            st.write("Your entered question was:")
            st.write(response['input'])
            st.write("The website provided has been embedded, strored in FAISS vectore db and converted as retriever chain to fetch context:")
            st.write(response['context'])
            st.write("The answer for your question is:")
            st.write(response['answer'])



if __name__ == "__main__":
    main()