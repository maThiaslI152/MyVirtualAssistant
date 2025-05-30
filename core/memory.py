# memory.py

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from config import REDIS_HOST, REDIS_PORT, REDIS_CHAT_PREFIX

def get_memory(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=f"{REDIS_CHAT_PREFIX}{session_id}",
        url=f"redis://{REDIS_HOST}:{REDIS_PORT}",
    )