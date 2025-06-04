# vectorstore.py

from langchain_community.vectorstores import Chroma
from .embedding import get_embedding_model
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAMESPACE


def get_vectorstore(embedding_model_name: str | None = None) -> Chroma:
    """Return a Chroma vector store configured with the optional embedding model."""
    embedding_model = None
    if embedding_model_name:
        embedding_model = get_embedding_model(embedding_model_name)

    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=CHROMA_COLLECTION_NAMESPACE,
        embedding_function=embedding_model,
    )
