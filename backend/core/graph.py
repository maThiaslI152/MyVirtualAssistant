from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from .models.chat_model import ChatState, Message
from .controllers.chat_controller import ChatController
from .presenters.chat_presenter import ChatPresenter
from .services.chat.redis import RedisService

class GraphState(TypedDict):
    chat_state: ChatState
    controller: ChatController
    presenter: ChatPresenter

async def process_message(state: GraphState) -> GraphState:
    """Process a message through the chat controller."""
    controller = state["controller"]
    chat_state = state["chat_state"]
    
    # Process the message
    updated_chat_state = await controller.process_message(chat_state)
    
    return {
        "chat_state": updated_chat_state,
        "controller": controller,
        "presenter": state["presenter"]
    }

def create_chat_graph(
    controller: ChatController,
    presenter: ChatPresenter,
    redis_service: RedisService
) -> StateGraph:
    """Create and configure the LangGraph for chat processing."""
    
    # Get Redis checkpointer
    checkpointer = redis_service.get_checkpointer()
    
    # Create the graph
    graph = StateGraph(GraphState, checkpointer=checkpointer)
    
    # Add nodes
    graph.add_node("process_message", process_message)
    
    # Set entry and exit points
    graph.set_entry_point("process_message")
    graph.set_finish_point("process_message")
    
    # Compile the graph
    return graph.compile()

def initialize_graph_state(
    controller: ChatController,
    presenter: ChatPresenter,
    initial_message: Message
) -> GraphState:
    """Initialize the graph state with required components."""
    return {
        "chat_state": {
            "messages": [initial_message],
            "context": {},
            "metadata": {}
        },
        "controller": controller,
        "presenter": presenter
    }