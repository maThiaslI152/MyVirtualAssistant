# Import Structure Documentation

## Overview

The project uses a modular import structure to organize services and components. This document outlines the import patterns and best practices for maintaining the codebase.

## Service Imports

### RAG Services
```python
# RAG Search Service
from ..services.rag.search import RAGSearchService

# Content Processing
from ..services.rag.processor import ContentProcessor

# Caching
from ..services.rag.cache import CacheService
```

### Chat Services
```python
# Core Chat Services
from ..services.chat.conversation import ConversationService
from ..services.chat.memory import MemoryService
from ..services.chat.enhancer import ChatEnhancer

# LLM and Embedding
from ..services.chat.llm import LLMService
from ..services.chat.embedding import EmbeddingService

# Redis Integration
from ..services.chat.redis import RedisService
```

### Processing Services
```python
# File Processing
from ..services.processing.file import FileProcessorService

# Image Processing
from ..services.processing.image import ImageProcessor

# Language Processing
from ..services.processing.language import LanguageProcessor

# Code Processing
from ..services.processing.code import CodeProcessor

# Voice Processing
from ..services.processing.voice import VoiceProcessor

# File Ingestion
from ..services.processing.ingest import process_file
```

### Search Services
```python
# Web Search
from ..services.search.web import WebSearchService

# Search History
from ..services.search.history import SearchHistoryService
```

## Import Best Practices

1. **Relative Imports**
   - Use relative imports (..) when importing from parent directories
   - Use absolute imports for core modules and third-party packages

2. **Service Organization**
   - Group related services in appropriate subdirectories
   - Use clear, descriptive names for service modules
   - Maintain consistent naming conventions

3. **Import Order**
   ```python
   # 1. Standard library imports
   import os
   import logging
   from typing import Dict, Any

   # 2. Third-party imports
   from fastapi import APIRouter
   from pydantic import BaseModel

   # 3. Local application imports
   from ..services.rag.search import RAGSearchService
   from ..services.chat.conversation import ConversationService
   ```

4. **Module Structure**
   - Each service module should have a clear, single responsibility
   - Use `__init__.py` files to expose public interfaces
   - Document module purposes and dependencies

## Common Patterns

### API Routes
```python
from fastapi import APIRouter
from ..services.rag.search import RAGSearchService
from ..services.processing.file import FileProcessorService

router = APIRouter()
```

### Core Components
```python
from ..services.chat.llm import LLMService
from ..services.chat.memory import MemoryService
from ..services.chat.embedding import EmbeddingService
```

### Processing Services
```python
from ..services.processing.file import FileProcessorService
from ..services.processing.image import ImageProcessor
from ..services.processing.language import LanguageProcessor
```

## Maintenance Guidelines

1. **Adding New Services**
   - Place new services in appropriate subdirectories
   - Update relevant `__init__.py` files
   - Document new services and their dependencies

2. **Refactoring**
   - Update all import statements when moving files
   - Maintain backward compatibility when possible
   - Update documentation to reflect changes

3. **Testing**
   - Ensure imports work in all environments
   - Test circular dependency scenarios
   - Verify import performance

## Troubleshooting

Common issues and solutions:

1. **Import Errors**
   - Check relative import paths
   - Verify `__init__.py` files exist
   - Ensure PYTHONPATH includes project root

2. **Circular Dependencies**
   - Use dependency injection
   - Restructure service dependencies
   - Consider using interfaces

3. **Module Not Found**
   - Verify file locations
   - Check import statements
   - Ensure virtual environment is activated 