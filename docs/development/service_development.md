# Service Development Guide

## Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment
- Required dependencies installed
- Development tools (IDE, Git, etc.)

### Setup
1. Clone the repository
2. Create and activate virtual environment
3. Install dependencies
4. Set up environment variables
5. Run development server

## Service Development

### 1. Creating a New Service

#### Basic Structure
```python
from typing import Dict, Any, Optional
import logging

class NewService:
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config

    async def process(self, data: Any) -> Dict[str, Any]:
        try:
            # Implementation
            return result
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise
```

#### Best Practices
- Use type hints
- Implement proper error handling
- Include logging
- Add docstrings
- Write tests

### 2. Service Integration

#### Adding to API
```python
from fastapi import APIRouter
from ..services.new_service import NewService

router = APIRouter()
service = NewService(config)

@router.post("/endpoint")
async def handle_request(data: Dict[str, Any]):
    return await service.process(data)
```

#### Service Dependencies
```python
class DependentService:
    def __init__(
        self,
        new_service: NewService,
        other_service: OtherService
    ):
        self.new_service = new_service
        self.other_service = other_service
```

### 3. Testing

#### Unit Tests
```python
import pytest
from ..services.new_service import NewService

def test_new_service():
    service = NewService(config)
    result = service.process(test_data)
    assert result == expected_result
```

#### Integration Tests
```python
async def test_service_integration():
    service = NewService(config)
    api_client = TestClient(app)
    response = await api_client.post("/endpoint", json=test_data)
    assert response.status_code == 200
```

### 4. Documentation

#### Service Documentation
```python
"""
New Service

This service handles specific functionality for the application.

Features:
- Feature 1
- Feature 2

Dependencies:
- Dependency 1
- Dependency 2

Usage:
    service = NewService(config)
    result = await service.process(data)
"""
```

#### API Documentation
```python
@router.post("/endpoint")
async def handle_request(
    data: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Process the request data.

    Args:
        data: The input data to process
        background_tasks: Background tasks handler

    Returns:
        Dict containing the processed results

    Raises:
        HTTPException: If processing fails
    """
```

## Development Workflow

### 1. Feature Development
1. Create feature branch
2. Implement changes
3. Write tests
4. Update documentation
5. Create pull request

### 2. Code Review
1. Review code changes
2. Check test coverage
3. Verify documentation
4. Test functionality
5. Approve changes

### 3. Deployment
1. Merge changes
2. Run integration tests
3. Deploy to staging
4. Verify functionality
5. Deploy to production

## Common Tasks

### 1. Adding New Endpoint
```python
@router.post("/new-endpoint")
async def new_endpoint(
    request: RequestModel,
    background_tasks: BackgroundTasks
) -> ResponseModel:
    try:
        result = await service.process(request)
        return ResponseModel(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 2. Implementing Background Task
```python
async def background_task(data: Dict[str, Any]):
    try:
        await service.process_async(data)
    except Exception as e:
        logger.error(f"Background task failed: {str(e)}")

@router.post("/endpoint")
async def handle_request(
    data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(background_task, data)
    return {"status": "processing"}
```

### 3. Error Handling
```python
class ServiceError(Exception):
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code
        super().__init__(self.message)

try:
    result = await service.process(data)
except ServiceError as e:
    raise HTTPException(status_code=e.code, detail=e.message)
```

## Performance Optimization

### 1. Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_function(param: str) -> Any:
    return expensive_operation(param)
```

### 2. Async Operations
```python
async def process_batch(items: List[Any]) -> List[Any]:
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks)
```

### 3. Resource Management
```python
async def process_with_timeout(
    data: Any,
    timeout: int = 30
) -> Any:
    try:
        return await asyncio.wait_for(
            service.process(data),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise ServiceError("Operation timed out")
```

## Security Considerations

### 1. Input Validation
```python
from pydantic import BaseModel, validator

class RequestModel(BaseModel):
    data: str

    @validator('data')
    def validate_data(cls, v):
        if not v.strip():
            raise ValueError("Data cannot be empty")
        return v
```

### 2. Authentication
```python
from fastapi import Depends, HTTPException
from ..auth import get_current_user

@router.post("/protected")
async def protected_endpoint(
    data: Dict[str, Any],
    user = Depends(get_current_user)
):
    if not user.has_permission("write"):
        raise HTTPException(status_code=403)
    return await service.process(data)
```

### 3. Rate Limiting
```python
from fastapi import Request
from ..utils.rate_limit import rate_limit

@router.post("/endpoint")
@rate_limit(limit=100, period=60)
async def rate_limited_endpoint(request: Request):
    return await service.process(request)
``` 