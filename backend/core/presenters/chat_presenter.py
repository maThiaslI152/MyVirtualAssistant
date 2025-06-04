from typing import Dict, Any
from ..models.chat_model import ChatState, ChatResponse, Message

class ChatPresenter:
    @staticmethod
    def format_response(state: ChatState) -> ChatResponse:
        """Format the chat state into a response suitable for API consumption."""
        latest_message = state["messages"][-1]
        
        return ChatResponse(
            message=latest_message,
            context=state["context"],
            metadata=state["metadata"]
        )
    
    @staticmethod
    def format_error(error: Exception) -> Dict[str, Any]:
        """Format error messages for API consumption."""
        return {
            "error": str(error),
            "type": error.__class__.__name__,
            "status": "error"
        }
    
    @staticmethod
    def format_chat_history(messages: list[Message]) -> Dict[str, Any]:
        """Format chat history for API consumption."""
        return {
            "messages": [msg.dict() for msg in messages],
            "count": len(messages)
        } 