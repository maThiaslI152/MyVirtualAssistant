"""
Test configuration and fixtures for the backend tests.
"""
import pytest
import asyncio
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ..main import app
from ..db.base import Base
from ..config import settings

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create async engine for tests
engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=True,
    future=True
)

# Create async session factory
async_session = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_db() -> AsyncGenerator:
    """Create test database and tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def db_session(test_db) -> AsyncGenerator[AsyncSession, None]:
    """Create a fresh database session for each test."""
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
def client() -> Generator:
    """Create a test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def test_settings():
    """Provide test settings."""
    return settings

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        async def get(self, key: str) -> str:
            return self.data.get(key)
        
        async def set(self, key: str, value: str, ttl: int = None) -> bool:
            self.data[key] = value
            return True
        
        async def delete(self, key: str) -> bool:
            if key in self.data:
                del self.data[key]
                return True
            return False
    
    return MockRedis()

@pytest.fixture
def mock_chroma():
    """Mock ChromaDB client for testing."""
    class MockChroma:
        def __init__(self):
            self.collections = {}
        
        def get_collection(self, name: str):
            if name not in self.collections:
                self.collections[name] = {}
            return self.collections[name]
        
        def create_collection(self, name: str):
            if name not in self.collections:
                self.collections[name] = {}
            return self.collections[name]
    
    return MockChroma() 