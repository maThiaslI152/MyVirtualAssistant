# graph.py

import time
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from embedding import vectorstore
from llm import llm
from memory import message_history

# Define structured state
class GraphState(TypedDict):
    input: str
    answer: str
    elapsed_ms: float

# RAG Node: check cache, fallback to LLM
def rag_node(state: GraphState) -> GraphState:
    user_input = state["input"].strip()
    start = time.time()

    cached = vectorstore.similarity_search(user_input, k=1)
    if cached:
        cached_doc = cached[0].page_content
        if "Q:" in cached_doc and user_input.lower() in cached_doc.lower():
            answer_start = cached_doc.find("A:")
            cached_answer = cached_doc[answer_start + 2:].strip() if answer_start != -1 else cached_doc
            return {
                "input": user_input,
                "answer": f"(cached) {cached_answer}",
                "elapsed_ms": round((time.time() - start) * 1000, 2)
            }

    # RAG fallback
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = [HumanMessage(content=f"Context:\n{context}"), HumanMessage(content=user_input)]
    response = llm.invoke(prompt)

    # Save to cache and memory
    vectorstore.add_texts([f"Q: {user_input}\nA: {response.content}"])
    message_history.add_user_message(user_input)
    message_history.add_ai_message(response.content)

    return {
        "input": user_input,
        "answer": response.content,
        "elapsed_ms": round((time.time() - start) * 1000, 2)
    }

# Graph Construction
workflow = StateGraph(GraphState)
workflow.add_node("rag", rag_node)
workflow.set_entry_point("rag")
workflow.add_edge("rag", END)
rag_graph = workflow.compile()
