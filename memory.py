# memory.py

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Redis configuration
SESSION_ID = "default-session"
REDIS_URL = "redis://localhost:6379"

# Redis-backed message history
message_history = RedisChatMessageHistory(
    session_id=SESSION_ID,
    url=REDIS_URL
)

# LangChain memory using Redis
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=message_history,
)
