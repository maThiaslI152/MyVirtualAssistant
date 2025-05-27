import os
import streamlit as st
import torch

# === Environment Setup ===
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Imports ===
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationChain

# === Streamlit UI ===
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("📘 Personal AI Assistant")

if st.sidebar.button("Check Apple MPS Support"):
    st.sidebar.write("MPS Available:", torch.backends.mps.is_available())

# === Embedding + Vector DB ===
CHROMA_DIR = "./chroma-data"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# === Chroma DB ===
vectorstore = Chroma(
    collection_name="langchain_collection",
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# === LLM + Memory ===
llm = ChatOpenAI(
    model="deepseek-r1-distill-qwen-7b",
    base_url="http://localhost:1234/v1",
    api_key="not-needed-for-local-testing",
    temperature=0.7,
    streaming=True,
    verbose=True,
    )
# === Short-Term Mem Ridis Cache ===
message_history = RedisChatMessageHistory(
    session_id="default",
    url="redis://localhost:6379",)
memory = ConversationBufferMemory(chat_memory=message_history, return_messages=True)

conversation_chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)

# === LangGraph Nodes ===
def summarize_node(state):
    user_input = state.get("input", "")
    response = conversation_chain.invoke({"input": user_input})
    return {
        "summary": response["response"],
        "history": memory.chat_memory.messages,
    }

# === State Graph ===
class GraphState(TypedDict):
    input: str
    summary: str

# === LangGraph Setup ===
workflow = StateGraph(GraphState)
workflow.add_node("summarizer", summarize_node)
workflow.set_entry_point("summarizer")
workflow.add_edge("summarizer", END)
summary_graph = workflow.compile()

# === Streamlit Input ===
user_input = st.text_area("Ask something or paste document text:")
if st.button("Run Assistant") and user_input:
    result = summary_graph.invoke({"input": user_input})
    st.subheader("📄 Summary")
    st.write(result["summary"])
