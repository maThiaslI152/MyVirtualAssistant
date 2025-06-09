import os
import re
from typing import Dict, Any, Optional, List, Set, Union
from pathlib import Path
from datetime import timedelta
import json
from pydantic_settings import BaseSettings
from pydantic import Field, validator, model_validator

class Settings(BaseSettings):
    # Project Settings
    PROJECT_NAME: str = "Personal Assistant"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    BASE_DIR: Path = Path(__file__).parent.parent
    TIMEZONE: str = "UTC"
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    API_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    TIMEOUT: int = 60
    API_TITLE: str = "Personal Assistant API"
    API_DESCRIPTION: str = "API for Personal Assistant with RAG capabilities"
    API_VERSION: str = "1.0.0"
    API_DOCS_URL: str = "/docs"
    API_REDOC_URL: str = "/redoc"
    API_OPENAPI_URL: str = "/openapi.json"
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./app.db"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 1800
    DATABASE_ISOLATION_LEVEL: str = "READ COMMITTED"
    DATABASE_SSL_MODE: str = "prefer"
    DATABASE_CONNECT_ARGS: Dict[str, Any] = Field(default_factory=dict)
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SSL: bool = False
    REDIS_POOL_SIZE: int = 10
    REDIS_POOL_TIMEOUT: int = 30
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    REDIS_RETRY_ON_TIMEOUT: bool = True
    REDIS_MAX_CONNECTIONS: int = 100
    
    # Cache Settings
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_MAX_MEMORY: int = 1024 * 1024 * 100  # 100MB
    CACHE_COMPRESSION_THRESHOLD: int = 1024  # 1KB
    CACHE_LEVELS: List[str] = ["memory", "redis"]
    CACHE_WARMUP_WORKERS: int = 4
    CACHE_PREFIX: str = "cache:"
    CACHE_KEY_SEPARATOR: str = ":"
    CACHE_DEFAULT_TIMEOUT: int = 300
    CACHE_IGNORE_ERRORS: bool = True
    
    # Search History Settings
    SEARCH_HISTORY_TTL: int = 30 * 24 * 3600  # 30 days
    MAX_HISTORY_PER_USER: int = 1000
    RECOMMENDATION_THRESHOLD: float = 0.7
    MAX_RECOMMENDATIONS: int = 5
    SEARCH_PATTERN_WINDOW: int = 100
    SEARCH_MIN_QUERY_LENGTH: int = 2
    SEARCH_MAX_QUERY_LENGTH: int = 200
    SEARCH_RESULTS_PER_PAGE: int = 10
    SEARCH_CACHE_TTL: int = 3600  # 1 hour
    
    # File Processing Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: Set[str] = {
        "pdf", "docx", "xlsx", "txt", "py", "js", "ts", "html", "css",
        "jpg", "jpeg", "png", "gif", "svg", "pptx", "rtf", "md"
    }
    OCR_LANGUAGES: List[str] = ["eng"]
    OCR_CONFIG: Dict[str, Any] = {
        "lang": "eng",
        "config": "--psm 3"
    }
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    TEMP_DIR: Path = BASE_DIR / "temp"
    FILE_CHUNK_SIZE: int = 1024 * 1024  # 1MB
    FILE_DELETE_AFTER_PROCESSING: bool = True
    FILE_MAX_CONCURRENT_UPLOADS: int = 5
    
    # Conversation Settings
    MAX_CONVERSATION_LENGTH: int = 100
    MAX_BRANCHES_PER_NODE: int = 5
    MAX_CONTEXT_LENGTH: int = 2000
    CONVERSATION_TTL: int = 7 * 24 * 3600  # 7 days
    MAX_MESSAGE_LENGTH: int = 4000
    CONVERSATION_CLEANUP_INTERVAL: int = 3600  # 1 hour
    CONVERSATION_BATCH_SIZE: int = 100
    CONVERSATION_MAX_RETRIES: int = 3
    
    # Frontend Settings
    FRONTEND_URL: str = "http://localhost:3000"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["*"]
    CORS_CREDENTIALS: bool = True
    CORS_MAX_AGE: int = 600
    STATIC_DIR: Path = BASE_DIR / "static"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    PASSWORD_MIN_LENGTH: int = 8
    PASSWORD_MAX_LENGTH: int = 100
    PASSWORD_REGEX: str = r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]{8,}$"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE: timedelta = timedelta(minutes=30)
    JWT_REFRESH_TOKEN_EXPIRE: timedelta = timedelta(days=7)
    JWT_SECRET_KEY: str = "your-jwt-secret-key-here"  # Change in production
    JWT_REFRESH_SECRET_KEY: str = "your-refresh-secret-key-here"  # Change in production
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[Path] = BASE_DIR / "logs" / "app.log"
    LOG_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 5
    LOG_ROTATION_INTERVAL: str = "midnight"
    LOG_COMPRESSION: bool = True
    LOG_FORMAT_JSON: bool = False
    
    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "chroma"
    VECTOR_DIMENSION: int = 1536
    VECTOR_METRIC: str = "cosine"
    VECTOR_INDEX_PATH: Path = BASE_DIR / "vectorstore" / "index"
    VECTOR_BATCH_SIZE: int = 100
    VECTOR_UPDATE_INTERVAL: int = 3600  # 1 hour
    VECTOR_CACHE_SIZE: int = 1000
    VECTOR_SEARCH_K: int = 5
    
    # ChromaDB Settings
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_NAME: str = "documents"
    CHROMA_PERSIST_DIRECTORY: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour
    RATE_LIMIT_STORAGE_URL: Optional[str] = None
    RATE_LIMIT_STRATEGY: str = "fixed-window"
    RATE_LIMIT_BLOCK_DURATION: int = 300  # 5 minutes
    
    # Background Tasks
    BACKGROUND_TASK_WORKERS: int = 4
    BACKGROUND_TASK_QUEUE_SIZE: int = 1000
    BACKGROUND_TASK_TIMEOUT: int = 300
    BACKGROUND_TASK_RETRY_COUNT: int = 3
    BACKGROUND_TASK_RETRY_DELAY: int = 60
    BACKGROUND_TASK_MAX_CONCURRENCY: int = 10
    
    # Email Settings
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_TLS: bool = True
    SMTP_SSL: bool = False
    EMAIL_FROM: Optional[str] = None
    EMAIL_TEMPLATES_DIR: Path = BASE_DIR / "email_templates"
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    METRICS_PATH: str = "/metrics"
    ENABLE_TRACING: bool = True
    TRACING_SERVICE_NAME: str = "personal-assistant"
    TRACING_HOST: str = "localhost"
    TRACING_PORT: int = 6831
    
    @validator("UPLOAD_DIR", "TEMP_DIR", "LOG_FILE", "VECTOR_INDEX_PATH", "STATIC_DIR", "TEMPLATES_DIR", "EMAIL_TEMPLATES_DIR")
    def create_directories(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None:
            v.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("ALLOWED_EXTENSIONS")
    def validate_extensions(cls, v: Set[str]) -> Set[str]:
        return {ext.lower() for ext in v}
    
    @validator("CORS_ORIGINS")
    def validate_cors_origins(cls, v: List[str]) -> List[str]:
        if "*" in v and len(v) > 1:
            return ["*"]
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v: str) -> str:
        if not v.startswith(("sqlite:///", "postgresql://", "mysql://")):
            raise ValueError("Invalid database URL")
        return v
    
    @validator("REDIS_PASSWORD")
    def validate_redis_password(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) < 8:
            raise ValueError("Redis password must be at least 8 characters")
        return v
    
    @model_validator(mode="after")
    @classmethod
    def validate_settings(cls, values):
        if values.ENVIRONMENT == "production":
            if values.SECRET_KEY == "your-secret-key-here":
                raise ValueError("SECRET_KEY must be set in production!")
        return values
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = "utf-8"
        json_loads = json.loads

# Create global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings

def update_settings(**kwargs) -> None:
    """Update settings dynamically."""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)

def get_env(key: str, default: Any = None) -> Any:
    """Get environment variable with fallback."""
    return os.getenv(key, default)

def get_database_url() -> str:
    """Get database URL with proper formatting."""
    url = settings.DATABASE_URL
    if url.startswith("sqlite"):
        return url
    return url.replace("postgres://", "postgresql://")

def get_redis_url() -> str:
    """Get Redis URL with proper formatting."""
    auth = f":{settings.REDIS_PASSWORD}@" if settings.REDIS_PASSWORD else ""
    protocol = "rediss" if settings.REDIS_SSL else "redis"
    return f"{protocol}://{auth}{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"

def get_cors_origins() -> List[str]:
    """Get CORS origins with proper formatting."""
    if "*" in settings.CORS_ORIGINS:
        return ["*"]
    return settings.CORS_ORIGINS

def get_jwt_settings() -> Dict[str, Any]:
    """Get JWT settings with proper formatting."""
    return {
        "algorithm": settings.JWT_ALGORITHM,
        "access_token_expire": settings.JWT_ACCESS_TOKEN_EXPIRE,
        "refresh_token_expire": settings.JWT_REFRESH_TOKEN_EXPIRE,
        "secret_key": settings.JWT_SECRET_KEY,
        "refresh_secret_key": settings.JWT_REFRESH_SECRET_KEY
    }

def get_email_settings() -> Dict[str, Any]:
    """Get email settings with proper formatting."""
    return {
        "host": settings.SMTP_HOST,
        "port": settings.SMTP_PORT,
        "user": settings.SMTP_USER,
        "password": settings.SMTP_PASSWORD,
        "tls": settings.SMTP_TLS,
        "ssl": settings.SMTP_SSL,
        "from_email": settings.EMAIL_FROM
    }

def get_monitoring_settings() -> Dict[str, Any]:
    """Get monitoring settings with proper formatting."""
    return {
        "metrics": {
            "enabled": settings.ENABLE_METRICS,
            "port": settings.METRICS_PORT,
            "path": settings.METRICS_PATH
        },
        "tracing": {
            "enabled": settings.ENABLE_TRACING,
            "service_name": settings.TRACING_SERVICE_NAME,
            "host": settings.TRACING_HOST,
            "port": settings.TRACING_PORT
        }
    }

def get_cache_settings() -> Dict[str, Any]:
    """Get cache settings with proper formatting."""
    return {
        "ttl": settings.CACHE_TTL,
        "max_memory": settings.CACHE_MAX_MEMORY,
        "compression_threshold": settings.CACHE_COMPRESSION_THRESHOLD,
        "levels": settings.CACHE_LEVELS,
        "prefix": settings.CACHE_PREFIX,
        "key_separator": settings.CACHE_KEY_SEPARATOR,
        "default_timeout": settings.CACHE_DEFAULT_TIMEOUT,
        "ignore_errors": settings.CACHE_IGNORE_ERRORS
    }

def get_vector_store_settings() -> Dict[str, Any]:
    """Get vector store settings with proper formatting."""
    return {
        "type": settings.VECTOR_STORE_TYPE,
        "dimension": settings.VECTOR_DIMENSION,
        "metric": settings.VECTOR_METRIC,
        "index_path": str(settings.VECTOR_INDEX_PATH),
        "batch_size": settings.VECTOR_BATCH_SIZE,
        "update_interval": settings.VECTOR_UPDATE_INTERVAL,
        "cache_size": settings.VECTOR_CACHE_SIZE,
        "search_k": settings.VECTOR_SEARCH_K
    } 