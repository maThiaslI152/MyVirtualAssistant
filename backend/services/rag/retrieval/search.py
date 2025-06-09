from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import uuid
import logging
import json
import hashlib
import os
import torch
import chromadb

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains.retrieval_qa.base import RetrievalQA

from ..core.rag_config import RAG_CONFIG, VECTORSTORE_CONFIG
from ..utils.helpers import generate_cache_key
from ..processing.processor import ContentProcessor
from ..processing.file import FileProcessorService
from .cache import RAGCache
from .web import WebSearchService
from core.models.chat_model import Message

class RAGSearchService:
    def __init__(
        self,
        openai_api_key: str,
        cache_service: RAGCache,
        content_processor: ContentProcessor,
        file_processor: FileProcessorService,
        web_search: WebSearchService,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.logger = logging.getLogger(__name__)
        self.openai_api_key = openai_api_key
        self.cache_service = cache_service
        self.content_processor = content_processor
        self.file_processor = file_processor
        self.web_search = web_search
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.timeout = timeout

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'mps' if torch.backends.mps.is_available() else 'cpu'}
        )

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=VECTORSTORE_CONFIG['collection_name'],
            embedding_function=self.embeddings,
            persist_directory=VECTORSTORE_CONFIG['persist_directory']
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4-turbo-preview",
            temperature=0.7
        )

        # Initialize QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )

    async def process_web_content(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process web content into chunks and create embeddings."""
        chunks = self.content_processor.chunk_text(content.get('content', ''))
        processed_chunks = []

        for chunk in chunks:
            # Check cache for embedding
            cached_embedding = await self.cache_service.get_cached_embedding(chunk)
            
            if cached_embedding:
                embedding = cached_embedding
            else:
                # Create new embedding
                embedding = self.content_processor.get_embedding(chunk)
                # Cache the embedding
                await self.cache_service.cache_embeddings(chunk, embedding)
            
            # Create metadata
            metadata = {
                'url': content.get('url'),
                'timestamp': content.get('timestamp'),
                'chunk_index': len(processed_chunks),
                'total_chunks': len(chunks),
                'word_count': len(chunk.split())
            }

            processed_chunks.append({
                'id': str(uuid.uuid4()),
                'content': chunk,
                'embedding': embedding,
                'metadata': metadata
            })

        return processed_chunks

    async def search(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search for relevant content using the query."""
        try:
            # Generate query embedding
            query_embedding = self.content_processor.get_embedding(query)
            
            # Search vector store
            results = self.vector_store.similarity_search_with_score(
                query,
                k=5
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': float(score)
                })
            
            return {
                'results': formatted_results,
                'query': query,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in search: {str(e)}")
            return None

    async def answer_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Answer a question using RAG."""
        try:
            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            
            return {
                'answer': result['result'],
                'sources': [doc.page_content for doc in result['source_documents']],
                'question': question,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in answer_question: {str(e)}")
            return None

    async def search_and_process(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search web, process content, and store in vector database."""
        # Check cache for search results
        cached_results = await self.cache_service.get_cached_search_results(query)
        if cached_results:
            return cached_results

        # Search web
        search_results = await self.web_search.search(query, num_results)
        
        # Process each result
        all_chunks = []
        for result in search_results:
            # Process content with enhanced features
            processed_content = self.content_processor.process_content(
                result.get('html', ''),
                result.get('url', ''),
                extract_summary=True,
                extract_entities=True,
                extract_keywords=True
            )
            
            if processed_content:
                chunks = await self.process_web_content(processed_content)
                all_chunks.extend(chunks)
                
                # Store in vector database
                docs = []
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk['content'],
                        metadata=chunk['metadata']
                    )
                    docs.append(doc)
                if docs:
                    self.vector_store.add_documents(docs)

        # Cache the results
        await self.cache_service.cache_search_results(query, all_chunks)
        
        return all_chunks

    async def retrieve_relevant_content(
        self,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant content from vector store."""
        # Get query embedding
        query_embedding = await self.content_processor.sentence_transformer.encode([query])
        
        # Search vector store
        results = self.vector_store.similarity_search(
            query_embedding,
            k=num_results
        )
        
        return results

    async def process_file(self, file_path: str) -> Dict[str, Any]:
        """Process a file and add it to the vector store."""
        try:
            # Process file
            processed_file = await self.file_processor.process_file(file_path)
            
            # Create documents from chunks
            documents = []
            for chunk in processed_file.chunks:
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        **chunk['metadata'],
                        'source': file_path,
                        'file_type': processed_file.file_type,
                        'processed_at': datetime.now().isoformat()
                    }
                )
                documents.append(doc)

            # Add to vector store
            self.vector_store.add_documents(documents)
            self.vector_store.persist()

            return {
                'status': 'success',
                'file_path': file_path,
                'file_type': processed_file.file_type,
                'chunks': len(documents),
                'metadata': processed_file.metadata
            }

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise

    async def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple files concurrently."""
        tasks = [self.process_file(file_path) for file_path in file_paths]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def search_files(
        self,
        query: str,
        file_types: Optional[List[str]] = None,
        num_results: int = 5
    ) -> Dict[str, Any]:
        """Search through processed files."""
        try:
            # Create filter for file types if specified
            filter_dict = None
            if file_types:
                filter_dict = {"file_type": {"$in": file_types}}

            # Search vector store
            results = self.vector_store.similarity_search(
                query,
                k=num_results,
                filter=filter_dict
            )

            # Process results
            processed_results = []
            for doc in results:
                processed_results.append({
                    'content': getattr(doc, 'page_content', ''),
                    'metadata': getattr(doc, 'metadata', {}),
                    'score': getattr(doc.metadata, 'score', 0) if hasattr(doc, 'metadata') else 0
                })

            return {
                'status': 'success',
                'query': query,
                'results': processed_results,
                'total': len(processed_results)
            }

        except Exception as e:
            self.logger.error(f"Error searching files: {str(e)}")
            raise

    async def rag_search(
        self,
        query: str,
        num_web_results: int = 5,
        num_retrieval_results: int = 5,
        allowed_domains: Optional[List[str]] = None,
        extract_summary: bool = True,
        extract_entities: bool = True,
        extract_keywords: bool = True,
        cache_ttl: Optional[int] = None,
        file_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Enhanced RAG search combining web search and file search."""
        try:
            # Generate cache key
            cache_key = hashlib.md5(
                json.dumps({
                    'query': query,
                    'num_web_results': num_web_results,
                    'num_retrieval_results': num_retrieval_results,
                    'allowed_domains': allowed_domains,
                    'file_types': file_types
                }).encode()
            ).hexdigest()

            # Check cache
            cached_result = await self.cache_service.get(cache_key)
            if cached_result:
                return cached_result

            # Perform web search
            web_results = await self.search_and_process(
                query,
                num_web_results
            )

            # Search through processed files
            file_results = await self.search_files(
                query,
                file_types,
                num_retrieval_results
            )

            # Combine results
            combined_results = {
                'web_results': web_results,
                'file_results': file_results['results'],
                'metadata': {
                    'web_results_count': len(web_results),
                    'file_results_count': len(file_results['results']),
                    'allowed_domains': allowed_domains,
                    'file_types': file_types,
                    'extraction_options': {
                        'summary': extract_summary,
                        'entities': extract_entities,
                        'keywords': extract_keywords
                    }
                }
            }

            # Cache results
            await self.cache_service.set(
                cache_key,
                combined_results,
                ttl=cache_ttl or self.cache_ttl
            )

            return combined_results

        except Exception as e:
            self.logger.error(f"Error in RAG search: {str(e)}")
            raise 