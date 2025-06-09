# Testing Documentation

## Overview

The project uses a comprehensive testing strategy with multiple levels of testing to ensure code quality and reliability.

## Test Structure

```
backend/tests/
├── unit/                    # Unit tests
│   ├── rag/                # RAG service unit tests
│   ├── chat/               # Chat service unit tests
│   ├── processing/         # Processing service unit tests
│   └── search/             # Search service unit tests
├── integration/            # Integration tests
│   ├── rag/               # RAG service integration tests
│   ├── chat/              # Chat service integration tests
│   ├── processing/        # Processing service integration tests
│   └── search/            # Search service integration tests
└── e2e/                   # End-to-end tests
    ├── rag/              # RAG service E2E tests
    ├── chat/             # Chat service E2E tests
    ├── processing/       # Processing service E2E tests
    └── search/           # Search service E2E tests
```

## Test Categories

### 1. Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Focus on specific functionality
- Fast execution

### 2. Integration Tests
- Test component interactions
- Use test database
- Test service integration
- Moderate execution time

### 3. End-to-End Tests
- Test complete workflows
- Use production-like environment
- Test user scenarios
- Longer execution time

## Running Tests

### Basic Test Execution
```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit
pytest tests/integration
pytest tests/e2e

# Run specific service tests
pytest tests/unit/rag
pytest tests/integration/chat
pytest tests/e2e/processing
```

### Test Options
```bash
# Run with coverage
pytest --cov=backend

# Run with verbose output
pytest -v

# Run specific test
pytest tests/unit/rag/test_search.py::test_search_function

# Run tests in parallel
pytest -n auto
```

## Test Fixtures

### Common Fixtures
```python
@pytest.fixture
async def db_session():
    """Database session fixture."""
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def client():
    """Test client fixture."""
    with TestClient(app) as test_client:
        yield test_client
```

### Mock Fixtures
```python
@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    return MockRedis()

@pytest.fixture
def mock_chroma():
    """Mock ChromaDB client."""
    return MockChroma()
```

## Writing Tests

### Unit Test Example
```python
def test_rag_search():
    """Test RAG search functionality."""
    service = RAGSearchService(mock_redis, mock_chroma)
    result = service.search("test query")
    assert result is not None
    assert "results" in result
```

### Integration Test Example
```python
async def test_chat_flow():
    """Test complete chat flow."""
    async with async_session() as session:
        service = ChatService(session)
        result = await service.process_message("Hello")
        assert result.response is not None
```

### E2E Test Example
```python
async def test_rag_workflow():
    """Test complete RAG workflow."""
    async with TestClient(app) as client:
        response = await client.post(
            "/api/rag/search",
            json={"query": "test query"}
        )
        assert response.status_code == 200
        assert "results" in response.json()
```

## Best Practices

### 1. Test Organization
- Group tests by functionality
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)
- Keep tests independent

### 2. Test Data
- Use fixtures for common data
- Clean up test data after tests
- Use meaningful test data
- Avoid hard-coded values

### 3. Mocking
- Mock external services
- Use appropriate mock levels
- Verify mock interactions
- Clean up mocks after tests

### 4. Assertions
- Use specific assertions
- Test edge cases
- Verify error conditions
- Check side effects

## Continuous Integration

### GitHub Actions
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-test.txt
      - name: Run tests
        run: |
          pytest --cov=backend
```

## Coverage Reports

### Generating Reports
```bash
# Generate HTML coverage report
pytest --cov=backend --cov-report=html

# Generate XML coverage report
pytest --cov=backend --cov-report=xml
```

### Coverage Configuration
```ini
[run]
source = backend
omit = 
    */tests/*
    */migrations/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
```

## Performance Testing

### Benchmark Tests
```python
def test_search_performance(benchmark):
    """Test search performance."""
    service = RAGSearchService()
    result = benchmark(service.search, "test query")
    assert result is not None
```

### Load Testing
```python
async def test_concurrent_requests():
    """Test concurrent request handling."""
    async with TestClient(app) as client:
        tasks = [
            client.post("/api/rag/search", json={"query": f"query {i}"})
            for i in range(100)
        ]
        results = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in results)
``` 