# backend/core/config.py

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env into os.environ
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
ENV = os.getenv("ENV", "dev")

# App config
APP_NAME = os.getenv("APP_NAME", "Owlynn AI Assistant")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8001))

# LLM config
LLM_API_URL = os.getenv("LLM_API")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3-14b")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))

# Embedding config
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Chroma vectorstore
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma/chroma")

# Postgres
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")