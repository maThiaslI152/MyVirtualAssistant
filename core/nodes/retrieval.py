# nodes/retrieval.py

from ..vectorstore import get_vectorstore


def retrieval_node(state: dict) -> dict:
    query = state["input"]
    # The embedding model name is populated by the previous node under
    # ``embedding_model_name``. Use that value if present.
    model_name = state.get("embedding_model_name")

    vectorstore = get_vectorstore(embedding_model_name=model_name)
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    docs = retriever.invoke(query)
    state["docs"] = docs
    return state
