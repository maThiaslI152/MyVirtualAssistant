from typing import List, Dict, Any, Generator, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from chromadb.config import Settings
import logging
from pathlib import Path
import torch

from ..core.rag_config import RAG_CONFIG, VECTORSTORE_CONFIG
from ..utils.helpers import ensure_directory

logger = logging.getLogger(__name__)

class RAGServiceError(Exception):
    """Base exception for RAG service errors"""
    pass

class DocumentProcessingError(RAGServiceError):
    """Exception raised when document processing fails"""
    pass

class RAGService:
    def __init__(
        self,
        collection_name: str = VECTORSTORE_CONFIG['collection_name'],
        persist_directory: str = VECTORSTORE_CONFIG['persist_directory'],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Ensure persist directory exists
        ensure_directory(persist_directory)

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'mps' if torch.backends.mps.is_available() else 'cpu'}
        )

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        # Initialize text splitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )

    def process_document(self, file_path: str) -> None:
        """Process a document and store it in the vector database."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {file_path}")

            # Load and split document
            loader = PyPDFLoader(str(file_path))
            pages = loader.load_and_split()
            
            # Split into chunks with metadata
            docs = self.text_splitter.split_documents(pages)
            
            # Store in ChromaDB with batch processing
            batch_size = 100
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                self.vector_store.add_documents(
                    documents=batch,
                    metadatas=[{
                        **doc.metadata,
                        "source": str(file_path),
                        "chunk_index": i + j
                    } for j, doc in enumerate(batch)]
                )
            
            logger.info(f"Successfully processed document: {file_path}")
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise DocumentProcessingError(f"Failed to process document: {e}")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents using the query."""
        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k
            )
            
            return [{
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            } for doc, score in results]
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return []

    def delete_document(self, source: str) -> None:
        """Delete a document and its chunks from the vector database."""
        try:
            # Get all documents with the given source
            results = self.vector_store.get(
                where={"source": source}
            )
            
            if results and results['ids']:
                # Delete the documents
                self.vector_store.delete(
                    ids=results['ids']
                )
                logger.info(f"Successfully deleted document: {source}")
            else:
                logger.warning(f"No documents found for source: {source}")
                
        except Exception as e:
            logger.error(f"Error deleting document {source}: {e}")
            raise RAGServiceError(f"Failed to delete document: {e}")

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.vector_store.delete_collection()
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info("Successfully cleared collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise RAGServiceError(f"Failed to clear collection: {e}") 