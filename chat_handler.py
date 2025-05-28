import os
import streamlit as st
from config import LIBRARY_DIR
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from app import llm, vectorstore, embeddings

def handle_chat(memory, message_history):
    user_input = st.chat_input("Ask something:")
    if not user_input:
        return

    st.chat_message("user").write(user_input)
    for fname in os.listdir(LIBRARY_DIR):
        if fname.lower() in user_input.lower():
            path = os.path.join(LIBRARY_DIR, fname)
            ext = fname.split('.')[-1].lower()

            if ext == "pdf":
                loader = PyPDFLoader(path)
            elif ext == "docx":
                loader = Docx2txtLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")

            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            response = llm.invoke(f"Summarize this document:\n{text}").content
            st.chat_message("ai").write(response)
            message_history.add_user_message(user_input)
            message_history.add_ai_message(response)
            st.stop()