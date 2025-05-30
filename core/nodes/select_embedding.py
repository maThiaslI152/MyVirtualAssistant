# nodes/select_embedding.py

from langchain_core.runnables import RunnableLambda

def select_embedding_node(state: dict) -> dict:
    user_input = state["input"]

    if "source code" in user_input or "Python" in user_input or "function" in user_input:
        model_name = "Qodo-Embed-1-1.5B"
    elif "abstract" in user_input or "research" in user_input or "semantic" in user_input:
        model_name = "jinaai/jina-embeddings-v3"
    else:
        model_name = "bge-en-icl"

    state["embedding_model_name"] = model_name
    return state

# make function runnable
select_embedding_node_runnable = RunnableLambda(select_embedding_node)