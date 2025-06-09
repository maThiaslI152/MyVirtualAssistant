# Owlynn AI Assistant

## Overview
Owlynn is a full-stack AI assistant project featuring a Next.js frontend and FastAPI backend with RAG (Retrieval-Augmented Generation) capabilities. The system integrates multiple AI technologies including ChromaDB for vector storage, sentence transformers for embeddings, and LLM integration for intelligent conversations.

## Architecture

### Backend (FastAPI)
- **API Framework**: FastAPI with async/await support
- **Vector Database**: ChromaDB for document embeddings and similarity search
- **Embeddings**: HuggingFace Transformers with `intfloat/multilingual-e5-large` model
- **LLM Integration**: Configured for streaming responses with `Qwen3-14b` model
- **Caching**: Redis for performance optimization
- **Database**: PostgreSQL for persistent storage
- **GPU Support**: MPS (Metal Performance Shaders) acceleration on Apple Silicon

### Frontend (Next.js)
- **Framework**: Next.js with TypeScript
- **UI Components**: Modern React components
- **Real-time Communication**: Chat interface with streaming support

### Infrastructure
- **Containerization**: Docker Compose for service orchestration
- **Services**: Frontend, Backend, PostgreSQL, Redis, ChromaDB
- **Development**: Local development with hot-reload support

## Current Status

### ✅ Completed Features
- [x] Docker Compose setup for all services
- [x] PostgreSQL database with initialization scripts
- [x] Redis caching service
- [x] ChromaDB vector database integration
- [x] RAG search service with document processing
- [x] Content processor with multiple NLP capabilities
- [x] File processor supporting multiple formats (PDF, DOCX, TXT, etc.)
- [x] MPS GPU acceleration for Apple Silicon
- [x] Sentence transformers for multilingual embeddings
- [x] Chat API with streaming support
- [x] Frontend build system and Docker integration

### 🔧 In Progress
- [ ] ChromaDB connection stabilization
- [ ] Backend service health checks
- [ ] LLM endpoint integration testing
- [ ] Frontend-backend API integration

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local backend development)
- Node.js 18+ (for local frontend development)

### Using Docker (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd Owlynn

# Start all services
docker-compose up --build

# Services will be available at:
# Frontend: http://localhost:3000
# ChromaDB: http://localhost:8000
# PostgreSQL: localhost:5432
# Redis: localhost:6379
```

### Local Development

#### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend (on port 8001 to avoid ChromaDB conflict)
uvicorn main:app --reload --port 8001
```

#### Frontend Setup
```bash
cd frontend

# Install dependencies
npm install

# Build the application
npm run build

# Start the frontend
npm start
```

## Configuration

### Environment Variables
Create a `.env` file in the backend directory:
```env
OPENAI_API_KEY=your-openai-api-key-here
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://owlynn:owlynnpass@localhost:5432/owlynn
```

### ChromaDB Settings
The system is configured to use ChromaDB with:
- REST API implementation
- Persistent storage in Docker volumes
- Multilingual embedding support
- Document collection management

## API Endpoints

### Core Endpoints
- `GET /api/ping` - Health check
- `POST /api/chat` - Chat with streaming support
- `POST /api/rag/search` - RAG search functionality
- `POST /api/rag/upload` - Document upload and processing

### RAG Features
- Web search integration
- Document similarity search
- Content summarization
- Entity extraction
- Keyword extraction
- Multi-format file support

## Development Notes

### GPU Acceleration
The system automatically detects and uses MPS (Metal Performance Shaders) on Apple Silicon Macs for faster model inference:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

### Dependencies Management
Key dependencies include:
- `fastapi`, `uvicorn` for API framework
- `chromadb[httpx]` for vector database
- `langchain-chroma`, `langchain-huggingface` for RAG
- `sentence-transformers` for embeddings
- `torch` with MPS support
- `transformers` for AI models

### Docker Services
- **PostgreSQL**: Database with custom initialization
- **Redis**: Caching and session management
- **ChromaDB**: Vector storage with REST API
- **Frontend**: Next.js application
- **Backend**: FastAPI application (run locally for development)

## Troubleshooting

### Common Issues
1. **Port Conflicts**: ChromaDB uses port 8000, run backend on 8001
2. **ChromaDB Connection**: Ensure Docker container is healthy
3. **GPU Memory**: MPS acceleration requires sufficient GPU memory
4. **Dependencies**: Use `pip install "chromadb[httpx]"` with quotes

### Health Checks
Check service status:
```bash
docker ps  # View running containers
docker logs owlynn-chroma  # Check ChromaDB logs
docker logs owlynn-postgres  # Check database logs
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license information here]

## Project Structure

```
Owlynn/
├── backend/                      # Backend application
│   ├── api/                     # API endpoints and routes
│   │   ├── rag.py              # RAG-specific endpoints
│   │   ├── conversation.py     # Conversation endpoints
│   │   └── ...
│   ├── core/                    # Core application components
│   │   ├── models/             # Data models
│   │   ├── schemas/            # Pydantic schemas
│   │   └── config/             # Configuration
│   ├── services/               # Business logic services
│   │   ├── rag/               # RAG-related services
│   │   │   ├── search.py      # RAG search service
│   │   │   ├── processor.py   # Content processor
│   │   │   └── cache.py       # Cache service
│   │   ├── chat/              # Chat-related services
│   │   │   ├── conversation.py # Conversation service
│   │   │   ├── memory.py      # Memory service
│   │   │   └── enhancer.py    # Chat enhancer
│   │   ├── processing/        # Content processing services
│   │   │   ├── file.py        # File processor
│   │   │   ├── image.py       # Image processor
│   │   │   └── code.py        # Code processor
│   │   └── search/            # Search-related services
│   │       ├── web.py         # Web search service
│   │       └── history.py     # Search history service
│   ├── db/                     # Database related code
│   │   ├── migrations/        # Database migrations
│   │   └── models/            # Database models
│   ├── vectorstore/           # Vector store implementation
│   ├── utils/                 # Utility functions
│   └── tests/                 # Backend tests
├── frontend/                   # Frontend application
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── pages/            # Page components
│   │   ├── services/         # Frontend services
│   │   └── utils/            # Frontend utilities
│   └── public/               # Static assets
├── docs/                      # Documentation
│   ├── api/                  # API documentation
│   ├── setup/               # Setup guides
│   └── architecture/        # Architecture docs
├── data/                     # Data storage
│   ├── uploads/             # Uploaded files
│   └── chroma/              # ChromaDB data
├── scripts/                  # Utility scripts
├── docker/                   # Docker configuration
│   ├── backend/             # Backend Dockerfile
│   └── frontend/            # Frontend Dockerfile
├── docker-compose.yml        # Docker compose configuration
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

# Project Name

## Overview
This project is a FastAPI application that integrates with ChromaDB for data storage and retrieval. It also includes a frontend for user interaction.

## Setup
1. **Clone the Repository**: Clone the repository to your local machine.
2. **Install Dependencies**: Navigate to the project directory and install the required dependencies using `pip install -r requirements.txt`.
3. **Run the Application**: Use the command `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8001` to start the FastAPI application.

## Dependencies
- FastAPI
- Uvicorn
- ChromaDB
- Newspaper3k
- Other dependencies listed in `requirements.txt`

## Running the Application
- **Backend**: Start the FastAPI application using the command provided in the setup section.
- **Frontend**: Ensure the frontend is configured to communicate with the backend API.

## Testing
- Use the Swagger UI at `http://localhost:8001/docs` to test the API endpoints.
