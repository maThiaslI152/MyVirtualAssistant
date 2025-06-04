from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class FileAttachment(BaseModel):
    filename: str
    content_type: str
    size: int
    file_id: str

class Message(BaseModel):
    role: str
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    file_attachment: Optional[FileAttachment] = None

class ChatState(TypedDict):
    messages: List[Message]
    context: Dict[str, Any]
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    message: Message
    context: Dict[str, Any]
    metadata: Dict[str, Any] 