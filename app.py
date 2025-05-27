import os
import streamlit as st
import torch
from typing import TypedDict
import time

# === Env Vars ===
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Imports ===
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_chroma import Chroma
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# === Streamlit UI ===
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("📘 Personal AI Assistant")

if st.sidebar.button("Check Apple MPS Support"):
    st.sidebar.write("MPS Available:", torch.backends.mps.is_available())

# === Embedding Wrapper ===
class MpsHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

# === Chroma + Embedding Setup ===
CHROMA_DIR = "./chroma-data"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
embeddings = MpsHuggingFaceEmbeddings(model=embedding_model)

vectorstore = Chroma(
    collection_name="langchain_collection",
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

# === LLM Setup ===
llm = ChatOpenAI(
    model="deepseek-r1-distill-qwen-7b",
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    temperature=0.7,
    streaming=True,
    verbose=True,
)

# === Optional Redis Memory (for future extension) ===
message_history = RedisChatMessageHistory(
    session_id="default",
    url="redis://localhost:6379",
)

# === LangGraph State ===
class GraphState(TypedDict):
    input: str
    answer: str
    elapsed_ms: float

# === LangGraph Node with Caching ===
def rag_node(state: GraphState) -> GraphState:
    user_input = state["input"].strip()
    start_time = time.time()

    # Step 1: Check for cached answer
    cached = vectorstore.similarity_search(user_input, k=1)
    if cached:
        cached_doc = cached[0].page_content
        if "Q:" in cached_doc and user_input.lower() in cached_doc.lower():
            answer_start = cached_doc.find("A:")
            cached_answer = cached_doc[answer_start + 2:].strip() if answer_start != -1 else cached_doc
            duration = round((time.time() - start_time) * 1000, 2)
            return {
                "input": user_input,
                "answer": f"(cached) {cached_answer}",
                "elapsed_ms": duration
            }

    # Step 2: Do retrieval + generate answer
    docs = vectorstore.similarity_search(user_input, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = [
        HumanMessage(content=f"Context:\n{context}"),
        HumanMessage(content=user_input)
    ]
    response = llm.invoke(prompt)

    # Step 3: Save Q&A to Chroma with ID (auto-persistent)
    vectorstore.add_texts(
        texts=[f"Q: {user_input}\nA: {response.content}"],
        ids=[f"cache-{hash(user_input)}"]
    )

    duration = round((time.time() - start_time) * 1000, 2)
    return {
        "input": user_input,
        "answer": response.content,
        "elapsed_ms": duration
    }

# === LangGraph Workflow ===
workflow = StateGraph(GraphState)
workflow.add_node("rag", rag_node)
workflow.set_entry_point("rag")
workflow.add_edge("rag", END)
rag_graph = workflow.compile()

# === Streamlit Input ===
user_input = st.text_area("Ask something:")
if st.button("Run Assistant") and user_input:
    result = rag_graph.invoke({"input": user_input})
    st.subheader("📄 Answer")

    answer_text = result["answer"].replace("(cached)", "").strip()
    elapsed = result.get("elapsed_ms", None)

    if result["answer"].startswith("(cached)"):
        st.markdown(f"🧠 **Cached answer** — answered in `{elapsed} ms`")
    else:
        st.markdown(f"✨ **LLM-generated answer** — completed in `{elapsed} ms`")

    st.write(answer_text)
