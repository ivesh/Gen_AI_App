## This project is about gen AI apps

# Gen AI Application

This repository contains a Gen AI application built using **LangChain**, **Streamlit**, and **Ollama's LLAMA2** model. It includes multiple functionalities such as chatbot history, language translation, and website Q&A features. The app leverages advanced AI tools to provide seamless interactions and solutions to user queries.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Contributing](#contributing)
- [License](#license)

---

## Project Structure

The project structure is organized as follows:


bash
Copy code
gen_venv_3.10/
│
├── main_page/
│   ├── pages/
│   │   ├── chatbot_history_llm.py      # Handles chatbot history functionality
│   │   ├── language_translator.py      # Implements language translation features
│   │   ├── website_q&a.py              # Provides Q&A functionality for websites
│   │   └── main_page.py                # Main entry point for the app
│
├── .env                                # Environment variables file
├── .gitignore                          # Git ignore file
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies


---

## Features

1. **Chatbot History**: Allows users to interact with an AI assistant and access previous chat history.
2. **Language Translation**: Supports seamless translation between multiple languages with different LangChain techniques.
3. **Website Q&A**: Enables users to ask questions based on specific website content and retrieves relevant answers.
4. **LLM-based Query Handling**: Uses LangChain with LLAMA2 and Groq for processing user queries intelligently.
5. **Document Embedding and Retrieval**: Processes website content into retrievable document chunks for efficient querying.

---

## Technologies Used

- **LangChain**: Provides a framework for building language model applications.
- **Ollama LLAMA2**: A lightweight, efficient language model for question-answering tasks.
- **Groq**: Accelerated inference for language models using Groq hardware.
- **Streamlit**: A framework for creating interactive web applications.
- **Python**: The core programming language for the application.
- **dotenv**: For managing environment variables.

---

## Setup Instructions

1. Clone this repository:
   ``` bash
   git clone <repository-url>
   cd gen_venv_3.10
2. Set up a virtual environment (Python 3.10+):

    ``` bash
        python3 -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

    ``` bash
    pip install -r requirements.txt
4. Configure environment variables in .env file:


    ```bash
    LANGCHAIN_API_KEY=your_langchain_api_key
    GROQ_API_KEY=your_groq_api_key
    LANGCHAIN_PROJECT=your_project_name
5. Run the app:
    ```bash


# Usage
- Navigate to the running Streamlit app.
- Use the input box to ask questions or interact with the AI assistant.
- Select different pages for functionalities like chatbot history, language translation, or website Q&A.

# File Descriptions
1. main_page.py
This file serves as the entry point for the application and includes:

- A simple Streamlit interface to accept user input.
- LangChain prompts for user queries, processed with the Ollama LLAMA2 model.
- An AI assistant to provide detailed responses to user questions.
2. chatbot_history_llm.py
- Handles chatbot functionality with session-based history. Key features include:

a. Persistent Chat History: Uses LangChain's ChatMessageHistory to store and retrieve chat conversations.
Role-Specific Responses: Allows users to set an AI "role" (e.g., Data Scientist) for custom interactions.
Message Trimming: Dynamically reduces the length of chat history while preserving context for efficient processing.
b. Groq Integration: Leverages Groq for fast and accurate LLM responses.
c. Streamlit Interface: A user-friendly interface for managing conversations and accessing history.
3. language_translator.py
- Provides advanced translation features using multiple LangChain techniques:

- Choice-Based Translation: Users can select between different translation methods:
System and human messages.
- Messages with output parsing.
- Messages with output parsing and direct model interaction.
- Generic template-based prompts.
- Streamlit UI: Includes input fields for source and target languages, as well as the text to be translated.
4. website_q&a.py
- Enables users to extract and query information from web pages. Features include:

- Document Loading and Splitting: Downloads web content and splits it into manageable chunks using RecursiveCharacterTextSplitter.
- Document Embedding: Creates embeddings with HuggingFace's all-MiniLM-L6-v2 model and stores them in a FAISS vector database.
- Context-Aware Q&A: Uses LangChain's retrieval chain to fetch relevant chunks and provide accurate answers to user queries.
- Streamlit UI: A straightforward interface for entering website URLs and user queries.
5. .env
Contains sensitive information such as API keys and project names. This file is essential for running the application and must not be shared publicly.

6. requirements.txt
Specifies all required Python packages, including:

- Streamlit
- LangChain
- dotenv
- FAISS
- HuggingFace models
- Ollama and Groq integrations.

License
This project is licensed under the MIT License.