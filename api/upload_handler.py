import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import LIBRARY_DIR
from app import embeddings

def handle_upload():
    uploaded_file = st.file_uploader("Upload a document for RAG (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        save_path = os.path.join(LIBRARY_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == "pdf":
            loader = PyPDFLoader(save_path)
        elif ext == "docx":
            loader = Docx2txtLoader(save_path)
        else:
            loader = TextLoader(save_path, encoding="utf-8")

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(docs)
        st.session_state.vectorstore.add_documents(docs)
        st.success("✅ File embedded and added to ChromaDB.")