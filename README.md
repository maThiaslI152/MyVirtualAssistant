Here’s a detailed and professional summary of your **Owlin** project suitable for `README.md`:

---

# 🦉 Owlin: Local AI Assistant for Smart Knowledge Management

**Owlin** is a local-first, privacy-focused AI assistant designed to help you think, learn, and organize. Inspired by tools like Notion and Obsidian, Owlin goes further by integrating **chat-based AI**, **file understanding**, **memory**, and **context branching** in one seamless app.

---

## ✅ Current Features

### 🧠 Chat Interface with Memory

* Streaming chat UI built with **Streamlit** (Vue.js version in progress).
* Supports **contextual responses** using:

  * User profile
  * Short-term memory (via **Redis**)
  * Long-term memory (via **ChromaDB**)
  * Session summaries and conversation pruning
* Smart **prompt injection** using profile, memory, and history.

### 📂 Multi-Type RAG System (Retrieval-Augmented Generation)

* RAG across various file types:

  * 📄 Text: `.txt`, `.md`, etc.
  * 📘 Documents: `.pdf`, `.docx`
  * 📊 Tables: `.csv`, `.xlsx`
  * 📷 Images: OCR and image captioning
  * 🧠 Code: Uses separate embedding route
* Smart routing to correct parser + embedding model.

### 🔍 Search & Caching System

* Semantic + exact match **response cache**:

  * Avoid redundant LLM calls
  * Faster and cheaper response reuse
* Vector logs stored in `.jsonl` format.

### ⚙️ Modular Backend

* Built with **FastAPI**
* Embedding powered by **HuggingFace** models:

  * `intfloat/multilingual-e5-large` (text)
  * `Qodo-Embed-1-1.5B` (code)
  * `jina-embeddings-v3` (optional backup)
* Apple Silicon optimized using **Metal (MPS)** via `llama-cpp-python`.

---

## 🛠️ In Progress

### 🌐 Frontend

* Building clean and mobile-friendly interface with:

  * **Vue 3 + Ionic + TailwindCSS**
  * Session-based branching UI like a **mind map**
  * Smart document list + summary viewer

### 🧩 Modularization

* Breaking backend into clean components:

  * Embedding pipeline
  * Memory manager
  * Chat agent controller
  * File manager

---

## 🚀 Planned Features

### 🧠 Advanced Memory Architecture

* Persistent **short-term memory** using Redis with token-based pruning + summarization
* Topic-aware **long-term memory** stored in ChromaDB
* Session-based **branching chat memory**, like a conversation tree

### 📚 Smart Document Library

* Sidebar interface to:

  * Upload + browse files
  * Auto-summarize and preview contents
  * Link file data to chat sessions via embedding

### 🔌 Tool Integration

* Web Search via plugin or headless browser
* Calculator, file explorer, and shell tools (local-first agents)
* Optional plugin execution system

### 🧾 Page Editing (Notion-like)

* Rich text pages stored in SQLite
* Markdown + block-level editing
* Bi-directional linking and embedding

### 🔒 Privacy and Local Control

* Fully offline, with **no third-party API calls**
* Compatible with Mac (Metal/MPS), Docker, and local GPU inference

---

## 🧪 Setup Instructions (Coming Soon)

Instructions to run with:

* Docker (for Redis + ChromaDB)
* Python (venv, FastAPI backend)
* Local inference using `llama-cpp-python`
* Vue 3 frontend (Ionic + Tailwind setup)

---

Let me know if you want this broken into actual Markdown file format, or if you want icons and badges added.
