from fastapi import APIRouter
import asyncpg
import redis.asyncio as aioredis

from backend.api.rag import router as rag_router
from api.conversation import router as conversation_router

router = APIRouter(prefix="/api")

@router.get("/ping")
async def root():
    return {"message": "Pong"}

@router.get("/health/redis")
async def health_redis():
    try:
        r = aioredis.from_url("redis://redis:6379/0")
        pong = await r.ping()
        return {"status": "ok" if pong else "fail"}
    except Exception as e:
        return {"status": "fail", "error": str(e)}

@router.get("/health/postgres")
async def health_postgres():
    try:
        conn = await asyncpg.connect("postgresql://owlynn:owlynnpass@postgres:5432/owlynndb")
        await conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "fail", "error": str(e)}

router.include_router(rag_router)
router.include_router(conversation_router)