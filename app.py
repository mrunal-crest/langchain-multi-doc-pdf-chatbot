import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
from main import create_chain, generate_response

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def chat_app():
    """
    Main function to create and run the Streamlit chat application.
    """
    try:
        # Streamlit app title
        st.title("Langchain Chat Bot")

        # Sidebar PDF selection
        pdf_select = st.sidebar.radio(
            "Please select your PDF",
            ["Airbus-Annual-Report-2023", "annualreport2223"],
            index=1
        )
        if pdf_select == "Airbus-Annual-Report-2023":
            pdf_path = "Airbus-Annual-Report-2023.pdf"
        elif pdf_select == "annualreport2223":
            pdf_path = "annualreport2223.pdf"

        # Initialize session state for storing chains and docsearches
        if "pdf_data" not in st.session_state:
            st.session_state.pdf_data = {}
        
        # Initialize session state for storing chat histories
        if "pdf_histories" not in st.session_state:
            st.session_state.pdf_histories = {}

        # Track the selected PDF in session state
        if "selected_pdf" not in st.session_state:
            st.session_state.selected_pdf = pdf_path

        # Check if the selected PDF has changed
        if st.session_state.selected_pdf != pdf_path:
            st.session_state.selected_pdf = pdf_path

        # Check if the selected PDF has been processed, if not, create chain and docsearch
        if pdf_path not in st.session_state.pdf_data:
            with st.spinner('Processing PDF and creating chain...'):
                docsearch, chain = create_chain(pdf_path)
                if docsearch and chain:
                    st.session_state.pdf_data[pdf_path] = {"docsearch": docsearch, "chain": chain}

        # Get the docsearch and chain for the selected PDF
        if pdf_path in st.session_state.pdf_data:
            docsearch = st.session_state.pdf_data[pdf_path]["docsearch"]
            chain = st.session_state.pdf_data[pdf_path]["chain"]
        else:
            docsearch, chain = None, None

        # Initialize chat history for the selected PDF
        if pdf_path not in st.session_state.pdf_histories:
            st.session_state.pdf_histories[pdf_path] = []

        # Display chat messages from history on app rerun
        for message in st.session_state.pdf_histories[pdf_path]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("What is up?"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.pdf_histories[pdf_path].append({"role": "user", "content": prompt})

            with st.spinner('Generating response...'):
                response = generate_response(docsearch, chain, prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)

            # Add assistant response to chat history
            st.session_state.pdf_histories[pdf_path].append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred in the chat app: {e}")

if __name__ == "__main__":
    chat_app()
