# Personal Assistant Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [API Reference](#api-reference)
5. [Configuration](#configuration)
6. [Development Guide](#development-guide)
7. [Deployment](#deployment)
8. [Security](#security)
9. [Troubleshooting](#troubleshooting)

## Overview

The Personal Assistant is a sophisticated application that combines Retrieval-Augmented Generation (RAG) with advanced conversation management and mind-map visualization capabilities. It's designed to provide an intelligent, context-aware assistant that can help users manage information, conduct searches, and maintain conversation history.

### Key Components

- **RAG System**: Combines retrieval and generation for context-aware responses
- **Conversation Management**: Tracks and manages conversation history with branching
- **Mind-Map Visualization**: Interactive visualization of conversation structures
- **File Processing**: Handles multiple file formats with advanced processing
- **Search History**: Tracks and analyzes search patterns

## Architecture

### Backend Architecture

```
backend/
├── api/              # API endpoints
├── services/         # Core services
├── core/            # Core functionality
└── config.py        # Configuration
```

#### Key Services

1. **RAG Service**
   - Handles search and retrieval
   - Manages context and responses
   - Integrates with vector store

2. **Search History Service**
   - Tracks user searches
   - Manages search patterns
   - Provides analytics

3. **File Processor Service**
   - Handles multiple file formats
   - Provides OCR capabilities
   - Manages file metadata

4. **Conversation Service**
   - Manages conversation history
   - Handles branching
   - Provides context management

### Frontend Architecture

```
frontend/
├── src/
│   ├── components/   # React components
│   ├── hooks/       # Custom hooks
│   ├── services/    # API services
│   └── utils/       # Utility functions
└── public/          # Static assets
```

## Features

### Core Features

1. **RAG-Powered Search**
   - Context-aware responses
   - Semantic search capabilities
   - Result ranking and filtering

2. **Conversation Management**
   - Branching conversations
   - Context preservation
   - History tracking

3. **Mind-Map Visualization**
   - Interactive visualization
   - Topic clustering
   - Relationship mapping

4. **File Processing**
   - Multiple format support
   - OCR capabilities
   - Metadata extraction

### Advanced Features

1. **Voice Input**
   - Speech-to-text conversion
   - Voice command support
   - Natural language processing

2. **Code Generation**
   - AI-powered code generation
   - Code analysis
   - Syntax highlighting

3. **URL Tracking**
   - URL analysis
   - Domain tracking
   - Pattern detection

4. **Analytics**
   - Usage statistics
   - Search patterns
   - Performance metrics

## API Reference

### RAG API

```python
POST /api/v1/rag/search
GET /api/v1/rag/history/{user_id}
GET /api/v1/rag/stats/{user_id}
```

### Conversation API

```python
POST /api/v1/conversation
GET /api/v1/conversation/{conversation_id}
GET /api/v1/conversation/{conversation_id}/branches
```

### File Processing API

```python
POST /api/v1/files/process
GET /api/v1/files/{file_id}
GET /api/v1/files/{file_id}/metadata
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

### Key Settings

1. **API Settings**
   - Port and host configuration
   - Rate limiting
   - CORS settings

2. **Database Settings**
   - Connection configuration
   - Pool settings
   - SSL configuration

3. **Redis Settings**
   - Connection details
   - Pool configuration
   - Timeout settings

4. **Security Settings**
   - JWT configuration
   - Password policies
   - Rate limiting

## Development Guide

### Setup Development Environment

1. **Backend Setup**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

### Running Tests

```bash
# Backend tests
pytest

# Frontend tests
npm test
```

### Code Style

- Backend: Follow PEP 8
- Frontend: Use ESLint and Prettier

## Deployment

### Production Deployment

1. **Backend Deployment**
   - Use Gunicorn or uWSGI
   - Configure SSL
   - Set up monitoring

2. **Frontend Deployment**
   - Build static files
   - Configure CDN
   - Set up caching

### Docker Deployment

```bash
docker-compose up -d
```

## Security

### Authentication

- JWT-based authentication
- Refresh token rotation
- Password hashing

### Authorization

- Role-based access control
- API key management
- Rate limiting

### Data Protection

- Input validation
- SQL injection prevention
- XSS protection

## Troubleshooting

### Common Issues

1. **API Connection Issues**
   - Check network configuration
   - Verify API endpoints
   - Check authentication

2. **Performance Issues**
   - Monitor resource usage
   - Check cache configuration
   - Optimize queries

3. **File Processing Issues**
   - Verify file permissions
   - Check file size limits
   - Validate file formats

### Debugging

1. **Backend Debugging**
   - Use logging
   - Enable debug mode
   - Check error logs

2. **Frontend Debugging**
   - Use browser dev tools
   - Check network requests
   - Monitor console logs

## Support

For additional support:
1. Check the [GitHub issues](https://github.com/yourusername/personal-assistant/issues)
2. Contact the development team
3. Join the community forum 