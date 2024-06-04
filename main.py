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

def generate_chain(pdf_path,prompt):
    try:
        pdf_reader = PdfReader(pdf_path)

        texts = ""
        for i, page in enumerate(pdf_reader.pages):
            text1 = page.extract_text()
            if text1:
                texts += text1
                
        token_count = estimate_tokens(texts)
        print(f"The estimated number of tokens in the PDF is: {token_count}")
            
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 32,
        length_function = len,
        )
        texts = text_splitter.split_text(texts)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        docs = docsearch.similarity_search(prompt)
        inputs = {'input_documents': docs, 'question': prompt}
        result = chain.invoke(input=inputs)
        return result['output_text']
    except Exception as e:
        print(f"An error occurred while generating the chain: {e}")
        return None
 
# generate_vector("D:/Project/Langchain ChatBot/langchain-multi-doc-pdf-chatbot/annualreport2223.pdf","https://www.singaporeair.com/saar5/pdf/Investor-Relations/Annual-Report/annualreport2223.pdf")
# generate_vector("D:/Project/Langchain ChatBot/langchain-multi-doc-pdf-chatbot/Airbus-Annual-Report-2023.pdf","https://www.airbus.com/sites/g/files/jlcbta136/files/2024-03/Airbus-Annual-Report-2023.pdf")
# generate_chain("D:/Project/Langchain ChatBot/langchain-multi-doc-pdf-chatbot/pdf_files_scan_create_reducefilesize.pdf","https://nett.umich.edu/sites/default/files/docs/pdf_files_scan_create_reducefilesize.pdf")


    # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY,model="text-embedding-ada-002")
    # pc = Pinecone(api_key=PINECONE_APIKEY)
    # index = pc.Index("test")

    # for i in texts:
    #     text_embeddings = embeddings.embed_query(text=i)
    #     print("inserted")
    #     vector = {  
    #             "id":str(uuid.uuid4()),
    #             "values":text_embeddings,
    #             "metadata":{
    #             "raw_text": i,
    #             "pdf_url": pdf_url
    #             }
    #         }

    #     index.upsert(vectors=[vector],namespace="ns1")