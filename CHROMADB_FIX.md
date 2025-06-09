# ChromaDB Fix Guide for Owlynn AI Assistant

## Problem Summary

The Owlynn project was experiencing ChromaDB connection issues due to:

1. **Deprecated API Configuration**: Using `chroma_api_impl="rest"` which is no longer supported
2. **Incorrect Client Settings**: Using outdated ChromaDB configuration parameters
3. **Version Compatibility**: Using unstable ChromaDB versions
4. **Environment Configuration**: Missing proper environment variable setup

## What Was Fixed

### 1. Docker Compose Configuration (`docker-compose.yml`)

**Before:**
```yaml
chroma:
  image: chromadb/chroma:latest
  environment:
    - CHROMA_API_IMPL=rest
    - CHROMA_SERVER_HOST=0.0.0.0
    # ... other deprecated settings
```

**After:**
```yaml
chroma:
  image: chromadb/chroma:0.5.15  # Stable version
  environment:
    - IS_PERSISTENT=TRUE
    - PERSIST_DIRECTORY=/chroma/chroma
    - ANONYMIZED_TELEMETRY=FALSE
    - ALLOW_RESET=TRUE
```

### 2. Python Client Configuration (`backend/services/rag_search_service.py`)

**Before:**
```python
self.vector_store = Chroma(
    collection_name="documents",
    embedding_function=self.embeddings,
    persist_directory="./data/chroma",
    client_settings=Settings(
        chroma_api_impl="rest",  # Deprecated!
        chroma_server_host="chroma",
        chroma_server_http_port=8000,
        # ...
    )
)
```

**After:**
```python
# Use HttpClient for proper server connection
self.chroma_client = chromadb.HttpClient(
    host=chroma_host,
    port=chroma_port,
    settings=chromadb.config.Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

self.vector_store = Chroma(
    collection_name="documents",
    embedding_function=self.embeddings,
    client=self.chroma_client
)
```

### 3. Dependencies Update (`backend/requirements.txt`)

**Updated ChromaDB version:**
```txt
# ChromaDB - stable version
chromadb>=0.5.15

# LangChain packages
langchain>=0.1.0
langchain-chroma>=0.1.0
```

### 4. Environment Configuration

**Created `backend/env.example`:**
```env
# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000

# OpenAI Configuration (Required)
OPENAI_API_KEY=your-openai-api-key-here
```

## Quick Fix Steps

### Option 1: Run the Automated Fix Script

```bash
# Run the comprehensive fix script
python fix_chroma_setup.py
```

### Option 2: Manual Steps

1. **Stop existing containers:**
   ```bash
   docker-compose down
   ```

2. **Pull the updated ChromaDB image:**
   ```bash
   docker pull chromadb/chroma:0.5.15
   ```

3. **Start ChromaDB container:**
   ```bash
   docker-compose up -d chroma
   ```

4. **Test the connection:**
   ```bash
   cd backend
   python test_chroma_connection.py
   ```

5. **Install/update Python dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

6. **Create environment file:**
   ```bash
   cp backend/env.example backend/.env
   # Edit backend/.env and add your OPENAI_API_KEY
   ```

## Testing the Fix

### 1. Test ChromaDB Container Health

```bash
# Check if container is running
docker ps | grep chroma

# Check container logs
docker logs owlynn-chroma

# Test HTTP endpoint
curl http://localhost:8000/api/v1/heartbeat
```

### 2. Test Python Client Connection

```bash
cd backend
python test_chroma_connection.py
```

Expected output:
```
Testing ChromaDB connection...
1. Testing ChromaDB server connectivity...
✅ ChromaDB server is responding!
2. Testing ChromaDB client connection...
✅ Successfully connected! Found 0 collections.
🎉 ChromaDB connection test completed successfully!
```

### 3. Test RAG Service Integration

```python
# Test in Python REPL
import os
os.environ["CHROMA_HOST"] = "localhost"
os.environ["CHROMA_PORT"] = "8000"

from backend.services.rag.search import RAGSearchService
# Should initialize without errors
```

## Common Troubleshooting

### Issue: "Connection refused" error

**Solution:**
```bash
# Check if port 8000 is available
lsof -i :8000

# Restart ChromaDB container
docker-compose restart chroma

# Wait for container to be fully ready
sleep 10
```

### Issue: "Unsupported Chroma API implementation" error

**Solution:** This was fixed by removing the deprecated `chroma_api_impl="rest"` configuration.

### Issue: Health check failures

**Solution:** Updated health check to use `wget` instead of `curl` (more reliable in container).

### Issue: Permission errors

**Solution:**
```bash
# Fix volume permissions
docker-compose down
docker volume rm owlynn_chroma_data
docker-compose up -d chroma
```

### Issue: Import errors

**Solution:**
```bash
# Update ChromaDB package
pip install --upgrade chromadb>=0.5.15
pip install --upgrade langchain-chroma>=0.1.0
```

## Verification Checklist

- [ ] ChromaDB container starts without errors
- [ ] Health check passes (container shows as "healthy")
- [ ] HTTP endpoint responds at `http://localhost:8000/api/v1/heartbeat`
- [ ] Python client can connect successfully
- [ ] Can create and query collections
- [ ] RAG service initializes without errors
- [ ] Environment variables are properly set

## Next Steps

1. **Set up your OpenAI API key** in `backend/.env`
2. **Start all services:** `docker-compose up -d`
3. **Test the full RAG pipeline** with document uploads
4. **Run the backend tests** to ensure everything works

## Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [ChromaDB Client/Server Mode](https://docs.trychroma.com/deployment)
- [LangChain ChromaDB Integration](https://python.langchain.com/docs/integrations/vectorstores/chroma)

---

**Note:** This fix addresses the core ChromaDB connection issues. For full RAG functionality, ensure you have a valid OpenAI API key configured in your environment. 