from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis import AsyncRedisSaver
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

# Set up AsyncRedisSaver (ensure Redis is running and accessible from this container)
checkpointer = AsyncRedisSaver.from_conn_string("redis://redis:6379")

# Create graph with async checkpointing
graph = StateGraph(ChatState, checkpointer=checkpointer)

graph.set_entry_point("start")
graph.add_node("start", llm_step)
graph.set_finish_point("start")  # You can also use END

# Compile the graph
chat_graph = graph.compile()
