"""
Services Module

This module contains all the service components of the application, organized into
specialized submodules:

1. RAG Services (rag/)
   - RAG search and processing
   - Content processing
   - Caching

2. Chat Services (chat/)
   - Conversation management
   - Memory handling
   - Chat enhancement

3. Processing Services (processing/)
   - File processing
   - Image processing
   - Language processing
   - Code processing

4. Search Services (search/)
   - Web search
   - Search history
   - Search analytics

Each submodule is designed to be modular and can be used independently or in
combination with other services in the application.
"""

from .rag import RAGSearchService
from .chat import ConversationService, MemoryService, ChatEnhancer
from .processing import FileProcessorService, ImageProcessor, LanguageProcessor, CodeProcessor
from .search import WebSearchService, SearchHistoryService

__all__ = [
    'RAGSearchService',
    'ConversationService',
    'MemoryService',
    'ChatEnhancer',
    'FileProcessorService',
    'ImageProcessor',
    'LanguageProcessor',
    'CodeProcessor',
    'WebSearchService',
    'SearchHistoryService'
]
