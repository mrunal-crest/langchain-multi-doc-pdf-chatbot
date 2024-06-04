import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from main import generate_chain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_APIKEY = os.getenv('PINECONE_APIKEY')

def generate_response(prompt, pdf_url):
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

def chat_app():
    try:
        # Streamlit app title
        st.title("Langchain Chat Bot")

        # Sidebar PDF selection
        pdf_select = st.sidebar.radio("Please select your PDF", ["Airbus-Annual-Report-2023", "annualreport2223", "Sample PDF"], index=0)
        if pdf_select == "Airbus-Annual-Report-2023":
            pdf_path = f"{os.getenv('AIRBUS_ANNUAL_REPORT')}"
        elif pdf_select == "Sample PDF":
            pdf_path = f"{os.getenv('SAMPLE_PDF')}"
        else:
            pdf_path = f"{os.getenv('ANNUAL_REPORT_2223')}"
        print(pdf_path)
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            
            with st.spinner('Generating response...'):
                result = generate_chain(pdf_path,prompt)
                response = f"Response: {result}"
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
    except Exception as e:
        st.error(f"An error occurred in the chat app: {e}")
        
if __name__ == "__main__":
    chat_app()
