# vectorstore.py

from langchain_community.vectorstores import Chroma
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAMESPACE


def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAMESPACE,
    )