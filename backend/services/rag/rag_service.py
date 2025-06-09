from typing import List, Dict, Any, Generator, Optional
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from chromadb.config import Settings
from chromadb.errors import ChromaDBError
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RAGServiceError(Exception):
    """Base exception for RAG service errors"""
    pass

class DocumentProcessingError(RAGServiceError):
    """Error during document processing"""
    pass

class QueryProcessingError(RAGServiceError):
    """Error during query processing"""
    pass

class RAGService:
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 8000,
        collection_name: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize RAG service with configurable parameters"""
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(allow_reset=True)
            )
            self.embedding_function = SentenceTransformerEmbeddings(
                model_name=embedding_model
            )
            self._setup_collection()
        except ChromaDBError as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise RAGServiceError(f"Failed to initialize ChromaDB client: {e}")

    def _setup_collection(self) -> None:
        """Initialize or get existing collection with error handling"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except ChromaDBError:
            try:
                self.collection = self.client.create_collection(self.collection_name)
            except ChromaDBError as e:
                logger.error(f"Failed to create collection: {e}")
                raise RAGServiceError(f"Failed to create collection: {e}")
        
        self.db = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def process_document(self, file_path: str) -> None:
        """Process and store document in vector store with improved error handling"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Document not found: {file_path}")

            # Load and split document
            loader = PyPDFLoader(str(file_path))
            pages = loader.load_and_split()
            
            # Split into chunks with metadata
            text_splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="\n"
            )
            docs = text_splitter.split_documents(pages)
            
            # Store in ChromaDB with batch processing
            batch_size = 100
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                self.collection.add(
                    documents=[doc.page_content for doc in batch],
                    metadatas=[{
                        **doc.metadata,
                        "source": str(file_path),
                        "chunk_index": i + j
                    } for j, doc in enumerate(batch)],
                    ids=[f"{file_path.stem}_{i + j}" for j in range(len(batch))]
                )
            
            logger.info(f"Successfully processed document: {file_path}")
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise DocumentProcessingError(f"Failed to process document: {e}")

    def get_context(
        self, 
        query: str, 
        k: int = 3,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> str:
        """Retrieve relevant context for query with filtering support"""
        try:
            docs = self.db.similarity_search(
                query, 
                k=k,
                filter=filter_criteria
            )
            return " ".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            raise QueryProcessingError(f"Failed to retrieve context: {e}")

    def get_system_message(self, context: str) -> str:
        """Generate system message with context and improved instructions"""
        return f"""You are an expert assistant helping users with their queries.
        Use the following context to provide accurate and relevant responses.

        CONTEXT:
        {context}

        Guidelines:
        1. Break down complex questions into simpler parts
        2. Use only information from the provided context
        3. Be specific and detailed in your responses
        4. If the context doesn't contain relevant information, say so
        5. Maintain a professional and helpful tone
        6. Cite sources when possible using the metadata
        7. Avoid hallucinations and stick to the provided context
        8. If uncertain, acknowledge the limitations of the available information
        """

    def get_user_message(self, query: str) -> str:
        """Format user query with improved structure"""
        return f"""Based on the provided context, please answer the following question:
        {query}

        Please ensure your response:
        1. Directly addresses the question
        2. Uses specific information from the context
        3. Maintains clarity and conciseness
        """

    def query(
        self, 
        query: str, 
        k: int = 3,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process query and return response with metadata"""
        try:
            context = self.get_context(query, k, filter_criteria)
            system_message = self.get_system_message(context)
            user_message = self.get_user_message(query)
            
            return {
                "context": context,
                "system_message": system_message,
                "user_message": user_message,
                "metadata": {
                    "query": query,
                    "k": k,
                    "filter_criteria": filter_criteria
                }
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise QueryProcessingError(f"Failed to process query: {e}")

    def stream_query(
        self, 
        query: str, 
        k: int = 3,
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream query processing results"""
        try:
            context = self.get_context(query, k, filter_criteria)
            system_message = self.get_system_message(context)
            user_message = self.get_user_message(query)
            
            yield {
                "type": "context",
                "content": context
            }
            
            yield {
                "type": "system_message",
                "content": system_message
            }
            
            yield {
                "type": "user_message",
                "content": user_message
            }
            
            yield {
                "type": "metadata",
                "content": {
                    "query": query,
                    "k": k,
                    "filter_criteria": filter_criteria
                }
            }
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            raise QueryProcessingError(f"Failed to stream query: {e}") 