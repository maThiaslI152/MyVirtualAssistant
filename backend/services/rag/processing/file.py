"""
File processing service for the RAG system.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

logger = logging.getLogger(__name__)

class FileProcessorService:
    """Service for processing various file types and preparing them for RAG."""
    
    def __init__(self, base_path: str = "./data"):
        """Initialize the file processor service.
        
        Args:
            base_path: Base path for storing processed files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # File type to loader mapping
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.csv': CSVLoader,
            '.md': UnstructuredMarkdownLoader
        }

    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a file and prepare it for RAG.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dict containing processed chunks and metadata
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get file extension and appropriate loader
            ext = file_path.suffix.lower()
            if ext not in self.loaders:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Load file content
            loader = self.loaders[ext](str(file_path))
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Process chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    'content': chunk.page_content,
                    'metadata': {
                        **chunk.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'file_path': str(file_path),
                        'file_type': ext[1:],  # Remove the dot
                        'processed_at': datetime.utcnow().isoformat()
                    }
                })
            
            return {
                'status': 'success',
                'file_path': str(file_path),
                'file_type': ext[1:],
                'chunks': processed_chunks,
                'metadata': {
                    'total_chunks': len(chunks),
                    'processed_at': datetime.utcnow().isoformat(),
                    'file_size': file_path.stat().st_size
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files concurrently.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processed file results
        """
        import asyncio
        tasks = [self.process_file(file_path) for file_path in file_paths]
        return await asyncio.gather(*tasks, return_exceptions=True) 