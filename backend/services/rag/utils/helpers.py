"""
Helper utilities for the RAG system.
"""

import hashlib
from typing import Any, Dict, List, Optional
from pathlib import Path

def generate_cache_key(query: str, **kwargs) -> str:
    """
    Generate a unique cache key for a query and its parameters.
    
    Args:
        query: The search query
        **kwargs: Additional parameters to include in the cache key
        
    Returns:
        A unique string key
    """
    # Combine query and kwargs into a single string
    key_parts = [query]
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    
    # Create a hash of the combined string
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def ensure_directory(path: Path) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: The directory path to ensure exists
    """
    path.mkdir(parents=True, exist_ok=True)

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: The text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text.strip()

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to split
        chunk_size: The size of each chunk
        overlap: The number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks 