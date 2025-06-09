from typing import Dict, Any, List, Optional
from ..models.chat_model import ChatState, Message, ChatResponse
from ..services.chat.llm import LLMService
from ..services.chat.memory import MemoryService
from ..services.chat.embedding import EmbeddingService

class ChatController:
    def __init__(
        self,
        llm_service: LLMService,
        memory_service: MemoryService,
        embedding_service: EmbeddingService
    ):
        self.llm_service = llm_service
        self.memory_service = memory_service
        self.embedding_service = embedding_service

    async def process_message(self, state: ChatState) -> ChatState:
        # Get relevant context from memory
        context = await self.memory_service.get_relevant_context(state["messages"])
        
        # Get embeddings for the latest message
        latest_message = state["messages"][-1]
        embeddings = await self.embedding_service.get_embeddings(latest_message.content)
        
        # Update state with context and embeddings
        state["context"] = context
        state["metadata"] = {"embeddings": embeddings}
        
        # Get LLM response
        response = await self.llm_service.generate_response(state)
        
        # Update state with response
        state["messages"].append(response)
        
        # Store in memory
        await self.memory_service.store_interaction(state)
        
        return state

    async def get_chat_history(self, chat_id: str) -> List[Message]:
        return await self.memory_service.get_chat_history(chat_id) 