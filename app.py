# app.py
# === Streamlit Chat Frontend ===
import streamlit as st
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage
from memory import message_history, memory
from graph import rag_graph

st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("📘 Personal AI Assistant")

# Check Apple MPS support
import torch
if st.sidebar.button("Check Apple MPS Support"):
    st.sidebar.write("MPS Available:", torch.backends.mps.is_available())

# === Load Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    for msg in message_history.messages:
        if isinstance(msg, HumanMessage):
            st.session_state.chat_history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            st.session_state.chat_history.append({"role": "assistant", "content": msg.content})

# === Render Chat ===
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === Handle Input ===
user_prompt = st.chat_input("Ask something...")
if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    result = rag_graph.invoke({"input": user_prompt})
    answer = result["answer"]
    elapsed = result["elapsed_ms"]

    label = f"🧠 *Cached* — `{elapsed}ms`" if answer.startswith("(cached)") else f"✨ *Generated* — `{elapsed}ms`"
    answer = answer.replace("(cached)", "").strip()

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(label)
