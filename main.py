import streamlit as st
import uuid
from PyPDF2 import PdfReader
from pinecone import Pinecone
from langchain.vectorstores import Weaviate, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_APIKEY = os.getenv('PINECONE_APIKEY')

def generate_vector(pdf_path,pdf_url):
    pdf_reader = PdfReader(pdf_path)

    text = ""
    for i, page in enumerate(pdf_reader.pages):
        text1 = page.extract_text()
        if text1:
            text += text1
            
    print(text,"******",len(text))
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 32,
    length_function = len,
    )
    
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-ada-002")

    for i in texts:
        text_embeddings = embeddings.embed_query(text=i)

    pc = Pinecone(api_key=PINECONE_APIKEY)
    index = pc.Index("test")
    vector = {  
            "id":str(uuid.uuid4()),
            "values":text_embeddings,
            "metadata":{
            "raw_text": text,
            "pdf_url": pdf_url
            }
        }

    index.upsert(vectors=[vector],namespace="ns1")
    
# generate_vector("D:/Project/Langchain ChatBot/langchain-multi-doc-pdf-chatbot/annualreport2223.pdf","https://www.singaporeair.com/saar5/pdf/Investor-Relations/Annual-Report/annualreport2223.pdf")
# generate_vector("D:/Project/Langchain ChatBot/langchain-multi-doc-pdf-chatbot/Airbus-Annual-Report-2023.pdf","https://www.airbus.com/sites/g/files/jlcbta136/files/2024-03/Airbus-Annual-Report-2023.pdf")
generate_vector("D:/Project/Langchain ChatBot/langchain-multi-doc-pdf-chatbot/pdf_files_scan_create_reducefilesize.pdf","https://nett.umich.edu/sites/default/files/docs/pdf_files_scan_create_reducefilesize.pdf")
