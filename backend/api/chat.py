from fastapi import APIRouter
from pydantic import BaseModel
from core.graph import chat_graph

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat(req: ChatRequest):
    result = chat_graph.invoke({"query": req.message, "history": []})
    return {"response": result["response"]}
