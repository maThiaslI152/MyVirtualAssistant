import os
import streamlit as st
import torch
from chromadb import PersistentClient

# Silence warnings
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Check MPS
if st.sidebar.button("Check Apple MPS Support"):
    st.sidebar.write("MPS Available:", torch.backends.mps.is_available())

# Model
llm = ChatOpenAI(
    model="deepseek-r1-distill-qwen-7b",
    base_url="http://localhost:1234/v1",
    api_key="none",
    temperature=0.7,
    max_tokens=4096
)

# Embeddings (MPS for Mac)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "mps"}
)

# Chroma
client = PersistentClient(path="./chroma-data")
vectorstore = Chroma(
    client=client,
    collection_name="langchain_collection",
    embedding_function=embeddings,
)

# Redis Chat Memory
session_id = "user-session"  # Can be replaced with dynamic session input
chat_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0",
    session_id=session_id,
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=chat_history,
    output_key="answer"
)

# Optional Tool
def search_engine(query: str) -> str:
    return f"Search Engine Result for: {query}"

tools = [
    Tool(
        name="Search Engine",
        func=search_engine,
        description="Useful for answering questions about current events or trends."
    )
]

# Prompt
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Conversational Chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# ReAct
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Streamlit Session State
if "memory" not in st.session_state:
    st.session_state.memory = memory

if "conversational_chain" not in st.session_state:
    st.session_state.conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=st.session_state.memory,
        return_source_documents=True,
        output_key="answer"
    )

# 🧠 COMMIT TO MEMORY BUTTON
if st.sidebar.button("🧠 Commit conversation to long-term memory"):
    history = chat_history.messages
    full_transcript = "\n".join(
        f"Human: {m.content}" if isinstance(m, HumanMessage) else f"AI: {m.content}"
        for m in history
    )
    # Summarize and embed to Chroma
    summarizer = load_summarize_chain(llm, chain_type="stuff")
    summary = summarizer.run([full_transcript])
    vectorstore.add_texts([summary])
    st.sidebar.success("✅ Conversation committed to ChromaDB.")

# UI
st.title("🔒 Offline AI Chat with RAG & MPS Acceleration")
user_input = st.text_input("Ask something:")

if user_input:
    response = st.session_state.conversational_chain.invoke({"question": user_input})
    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Sources")
    for i, doc in enumerate(response["source_documents"]):
        st.markdown(f"**Source {i+1}:**")
        st.code(doc.page_content)
