import streamlit as st
import uuid
from PyPDF2 import PdfReader
from pinecone import Pinecone
import tiktoken
# from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import os
from langchain_community.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_APIKEY = os.getenv('PINECONE_APIKEY')
def estimate_tokens(text, model="gpt-4"):
    try:
        # Initialize the tokenizer
        encoding = tiktoken.encoding_for_model(model)
        
        # Encode the text
        tokens = encoding.encode(text)
        
        # Return the number of tokens
        return len(tokens)
    except Exception as e:
        print(f"An error occurred while estimating tokens: {e}")
        return None
    
@st.cache_resource
def load_pdf(pdf_path):
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
    try:
        texts = load_pdf(pdf_path)
        if not texts:
            return None, None
        token_count = estimate_tokens(texts)
        print(f"The estimated number of tokens in the PDF is: {token_count}")
        st.sidebar.write(f"The estimated number of tokens in the PDF is: {token_count}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=32,
            length_function=len,
        )
        texts = text_splitter.split_text(texts)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        return docsearch, chain
    except Exception as e:
        st.error(f"An error occurred while creating the chain: {e}")
        return None, None

def generate_response(docsearch, chain, prompt):
    try:
        if not docsearch or not chain:
            raise ValueError("Invalid docsearch or chain.")
        docs = docsearch.similarity_search(prompt)
        inputs = {'input_documents': docs, 'question': prompt}
        result = chain.invoke(input=inputs)
        return result['output_text']
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")
        return "Error generating response."
    
def generate_OpenAI_response(prompt, pdf_url):
    client = OpenAI()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    text_embeddings = embeddings.embed_query(text=prompt)
    pc = Pinecone(api_key=PINECONE_APIKEY)
    index = pc.Index("test")
    query_response = index.query(
        namespace="ns1",
        vector=text_embeddings,
        filter={
            "pdf_url": {"$eq": pdf_url},
        },
        top_k=3,
        include_metadata=True
    )
    matches = query_response['matches']
    raw_texts = ""
    for i in matches:
        metadata = i.metadata
        raw_text = metadata['raw_text']
        raw_texts += raw_text
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. Please generate the proper result from this raw text for user input {raw_texts}"},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
