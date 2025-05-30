# embedding.py

import os
from langchain_huggingface import HuggingFaceEmbeddings
from core.config import CHROMA_PERSIST_DIR

def get_embedding_model(name: str = "bge-en-icl") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=name,
        cache_folder=os.path.join(CHROMA_PERSIST_DIR, "embeddings_cache"),
        model_kwargs={"device": "gpu"}
    )