from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from datetime import datetime
import json
import uuid
from backend.core.models.chat_model import Message, ChatResponse
from backend.core.controllers.chat_controller import ChatController
from backend.core.presenters.chat_presenter import ChatPresenter
from backend.core.graph import create_chat_graph, initialize_graph_state
from backend.services.chat.llm import LLMService
from backend.services.chat.memory import MemoryService
from backend.services.chat.embedding import EmbeddingService
from backend.services.chat.redis import RedisService
import httpx
import logging

router = APIRouter()

# Initialize services
llm_service = LLMService()
memory_service = MemoryService(
    redis_url="redis://redis:6379",
    pg_url="postgresql+asyncpg://owlynn:owlynnpass@postgres:5432/owlynndb"
)
embedding_service = EmbeddingService()
redis_service = RedisService()

# Initialize MCP components
controller = ChatController(llm_service, memory_service, embedding_service)
presenter = ChatPresenter()

# Create LangGraph
chat_graph = create_chat_graph(controller, presenter, redis_service)

async def call_llm(prompt: str) -> str:
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "Qwen3-14b",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4098,
        "stream": True
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

@router.post("/chat")
async def chat(
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
) -> Dict[str, Any]:
    try:
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Handle file upload if present
        file_data = None
        if file:
            file_content = await file.read()
            file_data = {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": len(file_content),
                "content": file_content
            }
            file_id = await memory_service.store_file(conversation_id, file_data)
        
        # Create message object
        msg = Message(
            role="user",
            content=message,
            timestamp=datetime.utcnow().timestamp(),
            metadata={"file_id": file_id} if file_data else None
        )
        
        # Store message in memory
        await memory_service.store_message(conversation_id, msg.dict())
        
        # Call LLM instead of using controller
        llm_response = await call_llm(message)
        
        # Create response message
        response = Message(
            role="assistant",
            content=llm_response,
            timestamp=datetime.utcnow().timestamp()
        )
        
        # Store response in memory
        await memory_service.store_message(conversation_id, response.dict())
        
        return {
            "conversation_id": conversation_id,
            "response": response.dict()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/chat/{conversation_id}/history")
async def get_chat_history(conversation_id: str) -> Dict[str, Any]:
    try:
        messages = await memory_service.get_conversation_messages(conversation_id)
        return {
            "conversation_id": conversation_id,
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/chat/{conversation_id}/files/{file_id}")
async def get_file(conversation_id: str, file_id: str) -> Dict[str, Any]:
    try:
        file_data = await memory_service.get_file(conversation_id, file_id)
        if not file_data:
            raise HTTPException(status_code=404, detail="File not found")
        return file_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup."""
    await redis_service.initialize()

@router.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown."""
    await redis_service.close()