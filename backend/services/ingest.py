import os
import fitz  # PyMuPDF
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

CHROMA_DIR = "/chroma/chroma"  # matches docker-compose volume

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def process_file(filename: str, content: bytes):
    # Step 1: Extract text
    if filename.endswith(".pdf"):
        text = extract_pdf(content)
    elif filename.endswith(".docx"):
        text = extract_docx(content)
    elif filename.endswith(".txt"):
        text = content.decode("utf-8")
    else:
        raise ValueError("Unsupported file type")

    # Step 2: Split text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])

    # Step 3: Embed + store in Chroma
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
    vectordb.add_documents(docs)
    vectordb.persist()

    return docs


def extract_pdf(content: bytes) -> str:
    with fitz.open(stream=content, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)

def extract_docx(content: bytes) -> str:
    temp_path = "/tmp/temp.docx"
    with open(temp_path, "wb") as f:
        f.write(content)
    return docx2txt.process(temp_path)
