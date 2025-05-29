# Graph.py

from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from nodes.select_embedding import select_embedding_node
from nodes.retrieval import retrieval_node

# Wrap nodes in a runnable
select_embedding = RunnableLambda(select_embedding_node)
retrieval = RunnableLambda(retrieval_node)

# Graph
graph = StateGraph()
graph.add_node("select_model", select_embedding)
graph.add_node("retrieve_docs", retrieval)

graph.set_entry_point("select_model")
graph.add_edge("select_model", "retrieve_docs")
graph.set_finish_point("retrieve_docs")

# Compile the graph
compiled_graph = graph.compile()