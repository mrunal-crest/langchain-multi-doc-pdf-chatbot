import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import tiktoken
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def estimate_tokens(text, model="gpt-3.5-turbo-16k’"):
    """
    Estimate the number of tokens in a given text for a specific model.

    Args:
        text (str): The input text to estimate tokens for.
        model (str): The model to use for token estimation (default is "gpt-3.5-turbo-16k’").

    Returns:
        int: The estimated number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"An error occurred while estimating tokens: {e}")
        return None

@st.cache_resource
def load_pdf(pdf_path):
    """
    Load and extract text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    try:
        pdf_reader = PdfReader(pdf_path)
        texts = ""
        for page in pdf_reader.pages:
            text1 = page.extract_text()
            if text1:
                texts += text1
        return texts
    except Exception as e:
        st.error(f"An error occurred while loading the PDF: {e}")
        return ""

@st.cache_resource
def create_chain(pdf_path):
    """
    Create a document search and QA chain from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        tuple: The document search object and QA chain.
    """
    try:
        texts = load_pdf(pdf_path)
        if not texts:
            return None, None
        token_count = estimate_tokens(texts)
        print(f"The estimated number of tokens in the PDF is: {token_count}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=32,
            length_function=len,
        )
        split_texts = text_splitter.split_text(texts)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(split_texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        return docsearch, chain
    except Exception as e:
        st.error(f"An error occurred while creating the chain: {e}")
        return None, None

def generate_response(docsearch, chain, prompt):
    """
    Generate a response to a given prompt using a document search and QA chain.

    Args:
        docsearch: The document search object.
        chain: The QA chain.
        prompt (str): The user prompt or question.

    Returns:
        str: The generated response.
    """
    try:
        if not docsearch or not chain:
            raise ValueError("Invalid docsearch or chain.")
         # Estimate the number of tokens in the prompt
        prompt_token_count = estimate_tokens(prompt)
        docs = docsearch.similarity_search(prompt)
        
        # Considering top 4 document from smiliary search
        inputs = {'input_documents': docs, 'question': prompt}
        result = chain.invoke(input=inputs)
        response_token_count = estimate_tokens(result['output_text'])
        if prompt_token_count and response_token_count:
            total_token_count = prompt_token_count + response_token_count
            print(f"The total estimated number of tokens used is: {total_token_count}, prompt tokens: {prompt_token_count} amd reponse tokens: {response_token_count}")
        return result['output_text']
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Error generating response."
