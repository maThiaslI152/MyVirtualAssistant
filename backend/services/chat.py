import requests
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DIR = "/chroma/chroma"
LLM_API_URL = "http://host.docker.internal:1234/v1/chat/completions"

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def ask_llm(query: str, history: list[str] = None) -> str:
    # Step 1: Retrieve docs from Chroma
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
    docs = vectordb.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Step 2: Build prompt for Qwen
    prompt = f"""You are Owlynn, a helpful assistant.
    
Use the following context to answer the user's question. Be concise and helpful. If context is unclear, do your best anyway.

Context:
{context}

User: {query}
Assistant:"""

    messages = [
        {"role": "system",
         "content": "You are Owlynn, an AI that helps answer questions using context from uploaded files."},
        {"role": "user", "content": prompt}
    ]

    # Step 3: Call LM Studio
    response = requests.post(
        LLM_API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen/qwen3-14b",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
        }
    )

    result = response.json()
    return result["choices"][0]["message"]["content"]