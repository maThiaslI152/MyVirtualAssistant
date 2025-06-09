from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import asyncio
from sqlalchemy import create_engine, Column, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import redis.asyncio as aioredis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import asyncpg
import uuid
from ..core.models.chat_model import Message, FileAttachment

Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    summary = Column(Text)
    metadata = Column(Text)  # JSON string

class Message(Base):
    __tablename__ = 'messages'
    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'))
    role = Column(String)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text)  # JSON string for file attachments, etc.

class MemoryService:
    def __init__(self, redis_url: str, pg_url: str):
        self.redis_url = redis_url
        self.pg_url = pg_url
        self.redis = None
        self.pg_pool = None
        
        # Initialize summarization chain
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
        # Start background task for session cleanup
        asyncio.create_task(self._cleanup_inactive_sessions())

    async def initialize(self):
        """Initialize Redis and PostgreSQL connections."""
        self.redis = await aioredis.from_url(self.redis_url)
        self.pg_pool = await asyncpg.create_pool(self.pg_url)

    async def close(self):
        """Close Redis and PostgreSQL connections."""
        if self.redis:
            await self.redis.close()
        if self.pg_pool:
            await self.pg_pool.close()

    async def store_message(self, conversation_id: str, message: Dict[str, Any]):
        """Store a message in Redis and PostgreSQL."""
        # Store in Redis for short-term memory
        message_key = f"conversation:{conversation_id}:messages"
        await self.redis.rpush(message_key, json.dumps(message))
        await self.redis.expire(message_key, 3600)  # Expire after 1 hour

        # Store in PostgreSQL for long-term memory
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO messages (id, conversation_id, role, content, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, 
            str(uuid.uuid4()),
            conversation_id,
            message['role'],
            message['content'],
            datetime.fromtimestamp(message['timestamp']),
            json.dumps(message.get('metadata', {}))
            )

    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get messages for a conversation from Redis or PostgreSQL."""
        # Try Redis first
        message_key = f"conversation:{conversation_id}:messages"
        messages = await self.redis.lrange(message_key, 0, -1)
        
        if messages:
            return [json.loads(msg) for msg in messages]
        
        # Fallback to PostgreSQL
        async with self.pg_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT role, content, timestamp, metadata
                FROM messages
                WHERE conversation_id = $1
                ORDER BY timestamp ASC
            """, conversation_id)
            
            return [{
                'role': row['role'],
                'content': row['content'],
                'timestamp': row['timestamp'].timestamp(),
                'metadata': json.loads(row['metadata']) if row['metadata'] else {}
            } for row in rows]

    async def summarize_conversation(self, conversation_id: str) -> str:
        """Summarize a conversation using the main LLM from LM Studio and cache the summary."""
        try:
            # Retrieve messages from Redis
            message_key = f"conversation:{conversation_id}:messages"
            messages = await self.redis.lrange(message_key, 0, -1)
            if not messages:
                return "No messages to summarize."
            
            # Combine messages into a single text
            combined_text = " ".join([json.loads(msg)['content'] for msg in messages])
            
            # Use the main LLM from LM Studio to generate a summary
            summary = await self.call_lm_studio_llm(combined_text, is_cache=True, stream=False)
            
            # Cache the summary in Redis
            await self.redis.set(f"conversation:summary:{conversation_id}", summary, ex=3600)
            
            # Store the summary in PostgreSQL for long-term memory
            async with self.pg_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE conversations SET summary = $1 WHERE id = $2
                """, summary, conversation_id)
            
            return summary
        except Exception as e:
            print(f"Error summarizing conversation {conversation_id}: {str(e)}")
            return "Error summarizing conversation."

    async def call_lm_studio_llm(self, text: str, is_cache: bool = False, stream: bool = False) -> str:
        """Call the Qwen3-14b model from LM Studio to generate a summary."""
        import aiohttp
        import json

        max_tokens = 2048 if is_cache else 4096
        prompt = f"Summarize the following text briefly: {text}" if not stream else f"Explain the following text in detail: {text}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:1234/v1/chat/completions",
                json={
                    "model": "Qwen3-14b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "stream": stream
                }
            ) as response:
                if response.status == 200:
                    if stream:
                        result = ""
                        async for line in response.content:
                            if line.startswith(b"data: "):
                                data = json.loads(line[6:])
                                if data["choices"][0]["finish_reason"] is not None:
                                    break
                                result += data["choices"][0]["delta"]["content"]
                        return result
                    else:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                else:
                    return "Error calling LM Studio."

    async def _cleanup_inactive_sessions(self) -> None:
        """Background task to check for and summarize inactive sessions."""
        while True:
            try:
                # Get all conversation keys
                keys = await self.redis.keys("conversation:last_activity:*")
                
                for key in keys:
                    conversation_id = key.split(":")[-1]
                    last_activity = await self.redis.get(key)
                    
                    if last_activity:
                        last_activity = datetime.fromisoformat(last_activity)
                        # Skip the main thread (conversation) from expiration
                        if conversation_id == "main":
                            continue
                        if datetime.utcnow() - last_activity > timedelta(hours=1):
                            # Session is inactive, summarize and move to LTM
                            await self.summarize_conversation(conversation_id)
                            await self.redis.delete(key)
                
            except Exception as e:
                print(f"Error in cleanup task: {str(e)}")
            
            # Wait for 5 minutes before next check
            await asyncio.sleep(300)

    async def store_file(self, conversation_id: str, file_data: Dict[str, Any]) -> str:
        """Store a file in PostgreSQL and return its ID."""
        file_id = str(uuid.uuid4())
        
        async with self.pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO files (id, conversation_id, filename, content_type, size, content)
                VALUES ($1, $2, $3, $4, $5, $6)
            """,
            file_id,
            conversation_id,
            file_data['filename'],
            file_data['content_type'],
            file_data['size'],
            file_data['content']
            )
        
        return file_id

    async def get_file(self, conversation_id: str, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a file from PostgreSQL."""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT filename, content_type, size, content
                FROM files
                WHERE id = $1 AND conversation_id = $2
            """, file_id, conversation_id)
            
            if not row:
                return None
            
            return {
                'filename': row['filename'],
                'content_type': row['content_type'],
                'size': row['size'],
                'content': row['content']
            } 