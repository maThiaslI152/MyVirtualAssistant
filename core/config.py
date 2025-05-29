#config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# ENV path
env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)

# LLM settings
llm_model = os.getenv("LLM_MODEL")
llm_temperature = float(os.getenv("LLM_TEMPERATURE"))
llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS"))
llm_top_p = float(os.getenv("LLM_TOP_P"))
llm_frequency_penalty = float(os.getenv("LLM_FREQUENCY_PENALTY"))
llm_presence_penalty = float(os.getenv("LLM_PRESENCE_PENALTY"))

# ChromaDB VectorDB settings
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR")
CHROMA_COLLECTION_NAMESPACE = os.getenv("CHROMA_COLLECTION_NAMESPACE")

# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_CHAT_PREFIX = os.getenv("REDIS_CHAT_PREFIX")

# File settings
FILE_PATH = os.getenv("UPLOAD_DIR")

# APP settings
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"