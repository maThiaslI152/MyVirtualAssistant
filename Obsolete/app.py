# Refactored AI Assistant with Summarization + Memory Pruning + Streaming UI
import os
import streamlit as st
import torch
import time
import json
from typing import TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from sentence_transformers.util import cos_sim

# === Env Setup ===
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === Streamlit Setup ===
memory_profile_path = "memory_profile.json"
def load_memory_profile():
    if os.path.exists(memory_profile_path):
        with open(memory_profile_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_memory_profile(profile):
    with open(memory_profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

profile_data = load_memory_profile()
st.set_page_config(page_title="AI Assistant", layout="wide")
if st.sidebar.button("Check Apple MPS Support"):
    st.sidebar.write("MPS Available:", torch.backends.mps.is_available())

# === Redis Memory ===
SESSION_ID = "default-session"
redis_url = "redis://localhost:6379"
message_history = RedisChatMessageHistory(session_id=SESSION_ID, url=redis_url)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=message_history,
)

# === Persistent Profile Sidebar ===
st.sidebar.markdown("### 🧠 Memory Profile")

# Load fields from profile_data
user_name = st.sidebar.text_input("Your Name", value=profile_data.get("user", ""))
style_pref = st.sidebar.text_input("Preferred Style", value=profile_data.get("prefers", ""))
project_context = st.sidebar.text_area("Project Context", value=profile_data.get("project", ""), height=80)

if st.sidebar.button("💾 Save Profile"):
    profile_data = {
        "user": user_name,
        "prefers": style_pref,
        "project": project_context
    }
    save_memory_profile(profile_data)
    st.sidebar.success("Memory profile updated.")

if st.sidebar.button("🗑️ Clear Profile"):
    profile_data = {}
    save_memory_profile(profile_data)
    st.sidebar.warning("Memory profile cleared.")

# === Clear Memory Buttons ===
if st.sidebar.button("🧹 Clear Short-Term Memory"):
    st.session_state.chat_history = []
    message_history.clear()
    st.sidebar.success("Short-term memory cleared.")

if st.sidebar.button("🗑️ Clear Vector Logs"):
    try:
        with open("vector_log.jsonl", "w", encoding="utf-8") as f:
            f.write("")
        st.sidebar.success("Vector logs cleared.")
    except Exception as e:
        st.sidebar.error(f"Error clearing logs: {e}")

if st.sidebar.checkbox("Show Vector Logs"):
    st.sidebar.markdown("### Vector Search Logs")
    try:
        with open("vector_log.jsonl", "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f.readlines()[-10:]]
        for log in logs:
            st.sidebar.markdown(f"**Query:** {log['query']}")
            st.sidebar.markdown(f"**Response:** {log['response'][:100]}...")
            st.sidebar.markdown(f"**Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(log['timestamp']))}")
            st.sidebar.markdown("---")
    except FileNotFoundError:
        st.sidebar.write("No logs yet.")

st.title("📘 Personal AI Assistant")

# === Custom Embedder ===
class MpsHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
embeddings = MpsHuggingFaceEmbeddings(model=embedding_model)
vectorstore = Chroma(
    collection_name="langchain_collection",
    persist_directory="./chroma-data",
    embedding_function=embeddings,
)
st.session_state.vectorstore = vectorstore

# === Owlin ===
llm = ChatOpenAI(
    model="deepseek-r1-distill-qwen-7b",
    base_url="http://localhost:1234/v1",
    api_key="none",
    temperature=0.7,
    max_tokens=4096,
)

# === Token Utility ===
def get_token_length(text: str) -> int:
    return len(text) // 4

def total_token_count(messages):
    return sum(get_token_length(m.content) for m in messages)

# === Load memory and maybe summarize ===
chat_history = memory.load_memory_variables({})["chat_history"]
MAX_TOKENS = 2000

if total_token_count(chat_history) > MAX_TOKENS:
    old_messages = chat_history[:-5]
    prompt = "Summarize this conversation:\n" + "\n".join([m.content for m in old_messages])
    summary = llm.invoke(prompt).content

    vectorstore.add_texts(
        texts=[summary],
        metadatas=[{"type": "chat_summary", "session_id": SESSION_ID, "timestamp": time.time()}]
    )

    recent_messages = chat_history[-5:]
    message_history.clear()
    message_history.add_messages(recent_messages)

# === Replay history to Streamlit UI ===
recent_messages = message_history.messages[-10:]
for msg in recent_messages:
    role = "user" if isinstance(msg, HumanMessage) else "ai"
    with st.chat_message(role):
        st.markdown(msg.content)

# Optional: expand to show full history
if st.sidebar.checkbox("Show Full Conversation History"):
    for msg in message_history.messages[:-10]:
        role = "user" if isinstance(msg, HumanMessage) else "ai"
        with st.chat_message(role):
            st.markdown(msg.content)

# === Chat input ===
user_input = st.chat_input("Ask something:")
if user_input:
    st.chat_message("user").write(user_input)

    # === Detect filename and summarize ===
    library_files = os.listdir("./library_files")
    for fname in library_files:
        if fname.lower() in user_input.lower():
            path = os.path.join("./library_files", fname)
            ext = fname.split('.')[-1].lower()

            if ext == "pdf":
                loader = PyPDFLoader(path)
            elif ext == "docx":
                loader = Docx2txtLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")

            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            response = llm.invoke(f"Summarize this document:\n{text}").content
            st.chat_message("ai").write(response)
            message_history.add_user_message(user_input)
            message_history.add_ai_message(response)
            st.stop()

    # === Cache Check ===
    cached = None
    try:
        with open("vector_log.jsonl", "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f]
            user_input_clean = user_input.strip().lower()

            # Step 1: try case-insensitive exact match
            for log in reversed(logs):
                if log["query"].strip().lower() == user_input_clean:
                    cached = log
                    break

            # Step 2: if no match, try embedding similarity
            if not cached:
                input_vec = embeddings.embed_query(user_input)
                for log in reversed(logs):
                    if "embedding" in log:
                        score = cos_sim([input_vec], [log["embedding"]])[0][0].item()
                        if score > 0.92:
                            cached = log
                            break
    except FileNotFoundError:
        pass

    if cached:
        response = cached["response"]
        st.chat_message("ai").write(response)
        st.chat_message("ai").markdown("**_Source: Cache_** — ⏱️ _cached_")
        message_history.add_user_message(user_input)
        message_history.add_ai_message(response)
        st.stop()
    else:
        start_time = time.time()
        source_label = "RAG + Owlin"

        retrieved_docs = vectorstore.similarity_search(user_input, k=3)
        summary_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) + "\n\n"
        past_turns = "\n".join([m.content for m in message_history.messages])
        profile_intro = json.dumps(profile_data, indent=2) if profile_data else ""
        full_prompt = profile_intro + "\n\n" + summary_text + past_turns + f"\nUser: {user_input}"

        ai_message = st.chat_message("ai")
        response_box = ai_message.empty()
        response = ""
        for chunk in llm.stream(full_prompt):
            content = chunk.content if hasattr(chunk, "content") else str(chunk)
            response += content
            response_box.write(response)

        elapsed = time.time() - start_time
        ai_message.markdown(f"**_Source: {source_label}_** — ⏱️ _{elapsed:.2f} sec_")

        message_history.add_user_message(user_input)
        message_history.add_ai_message(response)

        with open("vector_log.jsonl", "a", encoding="utf-8") as f:
            embedding_vec = embeddings.embed_query(user_input)
            f.write(json.dumps({
                "query": user_input,
                "response": response,
                "timestamp": time.time(),
                "source": source_label,
                "elapsed": elapsed,
                "embedding": embedding_vec
            }) + "\n")

# === File Upload and Embedding ===
uploaded_file = st.file_uploader("Upload a document for RAG (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import tempfile

    ext = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    if ext == "pdf":
        loader = PyPDFLoader(tmp_file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(tmp_file_path)
    else:
        loader = TextLoader(tmp_file_path, encoding='utf-8')

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Embed and add to vectorstore
    st.session_state.vectorstore.add_documents(docs)
    st.success("✅ File embedded and added to ChromaDB.")




# === File Library Viewer ===
if st.sidebar.button("📁 File Library"):
    st.session_state.view_file_library = True

if st.session_state.get("view_file_library"):
    st.title("📁 File Library")
    library_dir = "./library_files"
    os.makedirs(library_dir, exist_ok=True)
    files = os.listdir(library_dir)

    for fname in files:
        st.markdown(f"### 📄 {fname}")
        if st.button(f"Summarize {fname}", key=f"sum_{fname}"):
            from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            path = os.path.join(library_dir, fname)
            ext = fname.split('.')[-1].lower()

            if ext == "pdf":
                loader = PyPDFLoader(path)
            elif ext == "docx":
                loader = Docx2txtLoader(path)
            else:
                loader = TextLoader(path, encoding="utf-8")

            docs = loader.load()
            text = "\n".join([doc.page_content for doc in docs])
            summary = llm.invoke(f"Summarize this document:\n{text}").content
            st.markdown("#### 📌 Summary")
            st.markdown(summary)

            if st.checkbox("Preview content", key=f"preview_{fname}"):
                st.markdown("#### 📖 Full Content")
                st.markdown(text[:3000])
