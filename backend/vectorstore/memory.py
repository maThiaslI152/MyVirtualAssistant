# backend/vectorstore/memory.py

import json
import redis.asyncio as redis
from typing import List, Dict, Optional
from uuid import uuid4
from datetime import datetime
from summarizer import summarize_messages
from db import get_pg_session
from models import ArchivedSession
from sqlalchemy.exc import SQLAlchemyError

# Redis client
r = redis.Redis(host="localhost", port=6379, decode_responses=True)

SESSION_TTL_SECONDS = 3600  # 1 hour

def message(role: str, content: str) -> Dict:
    return {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }

def session_meta(title: str, parent_id: Optional[str]) -> Dict:
    return {
        "title": title or "Untitled",
        "created_at": datetime.utcnow().isoformat(),
        "parent_id": parent_id or "",
        "children": []
    }

async def _refresh_ttl(session_id: str):
    await r.expire(f"session:{session_id}:messages", SESSION_TTL_SECONDS)
    await r.expire(f"session:{session_id}:meta", SESSION_TTL_SECONDS)

async def create_session(title: str = "", parent_id: Optional[str] = None) -> str:
    session_id = str(uuid4())
    meta = session_meta(title, parent_id)

    await r.set(f"session:{session_id}:meta", json.dumps(meta), ex=SESSION_TTL_SECONDS)
    await r.delete(f"session:{session_id}:messages")
    await r.expire(f"session:{session_id}:messages", SESSION_TTL_SECONDS)

    if parent_id:
        parent_key = f"session:{parent_id}:meta"
        parent_data = await r.get(parent_key)
        if parent_data:
            parent = json.loads(parent_data)
            parent["children"].append(session_id)
            await r.set(parent_key, json.dumps(parent), ex=SESSION_TTL_SECONDS)

    return session_id

async def add_message(session_id: str, role: str, content: str):
    msg = message(role, content)
    await r.rpush(f"session:{session_id}:messages", json.dumps(msg))
    await _refresh_ttl(session_id)

async def get_memory(session_id: str) -> List[Dict]:
    await _refresh_ttl(session_id)
    raw = await r.lrange(f"session:{session_id}:messages", 0, -1)
    return [json.loads(msg) for msg in raw]

async def get_metadata(session_id: str) -> Dict:
    await _refresh_ttl(session_id)
    meta = await r.get(f"session:{session_id}:meta")
    return json.loads(meta) if meta else {}

async def clone_session(old_id: str, new_title: str = "") -> str:
    new_id = await create_session(new_title, parent_id=old_id)
    messages = await get_memory(old_id)
    if messages:
        await r.rpush(f"session:{new_id}:messages", *[json.dumps(m) for m in messages])
    await _refresh_ttl(new_id)
    return new_id

async def delete_session(session_id: str):
    await r.delete(f"session:{session_id}:meta")
    await r.delete(f"session:{session_id}:messages")

async def list_sessions() -> List[str]:
    keys = await r.keys("session:*:meta")
    return [k.split(":")[1] for k in keys]

async def redis_expiry_listener():
    pubsub = r.pubsub()
    await r.config_set("notify-keyspace-events", "Ex")
    await pubsub.psubscribe("__keyevent@0__:expired")

    async for msg in pubsub.listen():
        key = msg.get("data")
        if isinstance(key, bytes):
            key = key.decode()

        if key.startswith("session:") and key.endswith(":meta"):
            session_id = key.split(":")[1]
            await handle_expired_session(session_id)

async def handle_expired_session(session_id: str):
    try:
        raw_messages = await r.lrange(f"session:{session_id}:messages", 0, -1)
        messages = [json.loads(m) for m in raw_messages]
        meta = await r.get(f"session:{session_id}:meta")
        title = json.loads(meta)["title"] if meta else "Untitled"

        summary, tags = summarize_messages(messages)

        async with get_pg_session() as db:
            db.add(ArchivedSession(
                session_id=session_id,
                title=title,
                summary=summary,
                tags=tags,
                full_history=json.dumps(messages)
            ))
            await db.commit()

    except SQLAlchemyError as e:
        print(f"[DB Error] Failed to archive {session_id}: {e}")
    except Exception as e:
        print(f"[Error] Session {session_id} archiving failed: {e}"