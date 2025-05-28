import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from config import LIBRARY_DIR
from app import llm

def render_library():
    if st.sidebar.button("📁 File Library"):
        st.session_state.view_file_library = True

    if st.session_state.get("view_file_library"):
        st.title("📁 File Library")
        files = os.listdir(LIBRARY_DIR)

        for fname in files:
            st.markdown(f"### 📄 {fname}")
            if st.button(f"Summarize {fname}", key=f"sum_{fname}"):
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
                summary = llm.invoke(f"Summarize this document:\n{text}").content
                st.markdown("#### 📌 Summary")
                st.markdown(summary)

                if st.checkbox("Preview content", key=f"preview_{fname}"):
                    st.markdown("#### 📖 Full Content")
                    st.markdown(text[:3000])