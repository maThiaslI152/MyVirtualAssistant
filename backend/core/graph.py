from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import RedisCheckpoint
from typing import TypedDict
from services.chat import ask_llm

# Define the shape of the state
class ChatState(TypedDict):
    query: str
    history: list[str]
    response: str

# LangGraph step
def llm_step(state: ChatState) -> ChatState:
    query = state["query"]
    history = state.get("history", [])
    response = ask_llm(query, history)
    return {"query": query, "response": response, "history": history + [query, response]}

# Optional: Attach RedisCheckpoint (requires Redis running)
# checkpoint = RedisCheckpoint.from_url("redis://localhost:6379")
# graph = StateGraph(ChatState, checkpoint=checkpoint)

# If no checkpointing needed:
graph = StateGraph(ChatState)

graph.set_entry_point("start")
graph.add_node("start", llm_step)
graph.set_finish_point("start")  # or use END

# Compile the graph
chat_graph = graph.compile()
