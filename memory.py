import json
import torch
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
import streamlit as st

PROFILE_PATH = "memory_profile.json"
REDIS_URL = "redis://localhost:6379"

def load_memory_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_memory_profile(profile):
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

def init_memory():
    st.sidebar.markdown("### 🧠 Memory Profile")
    profile_data = load_memory_profile()
    user_name = st.sidebar.text_input("Your Name", value=profile_data.get("user", ""))
    style_pref = st.sidebar.text_input("Preferred Style", value=profile_data.get("prefers", ""))
    project_context = st.sidebar.text_area("Project Context", value=profile_data.get("project", ""), height=80)

    if st.sidebar.button("💾 Save Profile"):
        profile_data = {"user": user_name, "prefers": style_pref, "project": project_context}
        save_memory_profile(profile_data)
        st.sidebar.success("Memory profile updated.")

    if st.sidebar.button("🗑️ Clear Profile"):
        save_memory_profile({})
        st.sidebar.warning("Memory profile cleared.")

    message_history = RedisChatMessageHistory(session_id="default-session", url=REDIS_URL)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, chat_memory=message_history)
    return memory, message_history