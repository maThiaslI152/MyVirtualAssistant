"""
Configuration settings for the RAG (Retrieval-Augmented Generation) system.
"""

from typing import Dict, Any
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
VECTORSTORE_DIR.mkdir(exist_ok=True)

# Docker Configuration
DOCKER_CONFIG = {
    "chroma_host": os.getenv("CHROMA_HOST", "localhost"),
    "chroma_port": int(os.getenv("CHROMA_PORT", "8000")),
    "chroma_settings": {
        "allow_reset": True,
        "anonymized_telemetry": False
    }
}

# RAG Configuration
RAG_CONFIG: Dict[str, Any] = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_results": 5,
    "similarity_threshold": 0.7,
    "cache_ttl": 3600,  # 1 hour in seconds
    "vectorstore": {
        "name": "chroma",
        "collection_name": "documents",
        "embedding_model": "all-MiniLM-L6-v2",
    },
    "search": {
        "max_results": 5,
        "similarity_threshold": 0.7,
    },
    "processing": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_tokens": 2000,
    }
}

# Cache Configuration
CACHE_CONFIG: Dict[str, Any] = {
    "ttl": 3600,  # 1 hour in seconds
    "max_size": 1000,  # Maximum number of cached items
    "storage_path": str(CACHE_DIR),
}

# Vector Store Configuration
VECTORSTORE_CONFIG: Dict[str, Any] = {
    "name": "chroma",
    "collection_name": "documents",
    "embedding_model": "all-MiniLM-L6-v2",
    "persist_directory": str(VECTORSTORE_DIR),
    "docker": DOCKER_CONFIG
} 