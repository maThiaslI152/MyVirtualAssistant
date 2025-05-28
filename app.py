import streamlit as st
from config import init_settings
from memory import init_memory
from upload_handler import handle_upload
from library_view import render_library
from chat_handler import handle_chat

# === Setup ===
init_settings()
memory, message_history = init_memory()

# === UI ===
st.title("📘 Personal AI Assistant")

handle_upload()
handle_chat(memory, message_history)
render_library()