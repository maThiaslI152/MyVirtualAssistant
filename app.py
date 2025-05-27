import os
import streamlit as st
import torch
import time
from typing import TypedDict

# === Env ===
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Imports ===
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# === Streamlit Setup ===
st.set_page_config(page_title="AI Assistant", layout="wide")

if st.sidebar.button("Check Apple MPS Support"):
    st.sidebar.write("MPS Available:", torch.backends.mps.is_available())

st.title("📘 Personal AI Assistant")

# === Redis Setup ===
SESSION_ID = "default-session"
redis_url = "redis://localhost:6379"

message_history = RedisChatMessageHistory(
    session_id=SESSION_ID,
    url=redis_url
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=message_history,
)

# === Embedder ===
class MpsHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

# === Vectorstore ===
CHROMA_DIR = "./chroma-data"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
embeddings = MpsHuggingFaceEmbeddings(model=embedding_model)

vectorstore = Chroma(
    collection_name="langchain_collection",
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# === LLM ===
llm = ChatOpenAI(
    model="deepseek-r1-distill-qwen-7b",
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.7,
    streaming=True,
    verbose=True,
)

# === LangGraph ===
class GraphState(TypedDict):
    input: str
    answer: str
    elapsed_ms: float

def rag_node(state: GraphState) -> GraphState:
    user_input = state["input"].strip()
    start = time.time()

    # Cache check
    cached = vectorstore.similarity_search(user_input, k=1)
    if cached:
        cached_doc = cached[0].page_content
        if "Q:" in cached_doc and user_input.lower() in cached_doc.lower():
            answer_start = cached_doc.find("A:")
            cached_answer = cached_doc[answer_start + 2:].strip() if answer_start != -1 else cached_doc
            elapsed = round((time.time() - start) * 1000, 2)
            return {
                "input": user_input,
                "answer": f"(cached) {cached_answer}",
                "elapsed_ms": elapsed
            }

    # RAG flow
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = [
        HumanMessage(content=f"Context:\n{context}"),
        HumanMessage(content=user_input)
    ]
    response = llm.invoke(prompt)

    # Save to cache
    vectorstore.add_texts([f"Q: {user_input}\nA: {response.content}"])

    # Save to Redis
    message_history.add_user_message(user_input)
    message_history.add_ai_message(response.content)

    elapsed = round((time.time() - start) * 1000, 2)
    return {
        "input": user_input,
        "answer": response.content,
        "elapsed_ms": elapsed
    }

workflow = StateGraph(GraphState)
workflow.add_node("rag", rag_node)
workflow.set_entry_point("rag")
workflow.add_edge("rag", END)
rag_graph = workflow.compile()

# === Streamlit Chat UI ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load Redis memory into UI (on first launch)
if not st.session_state.chat_history:
    for msg in message_history.messages:
        if isinstance(msg, HumanMessage):
            st.session_state.chat_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            st.session_state.chat_history.append({"role": "assistant", "content": msg.content})

# Show previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_prompt = st.chat_input("Ask something...")
if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    result = rag_graph.invoke({"input": user_prompt})
    answer = result["answer"]
    elapsed = result["elapsed_ms"]

    if answer.startswith("(cached)"):
        label = f"🧠 *Cached* — `{elapsed}ms`"
        answer = answer.replace("(cached)", "").strip()
    else:
        label = f"✨ *Generated* — `{elapsed}ms`"

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(label)
