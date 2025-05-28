import os
import streamlit as st

LIBRARY_DIR = "./library_files"
SESSION_ID = "default-session"

def init_settings():
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    st.set_page_config(page_title="AI Assistant", layout="wide")