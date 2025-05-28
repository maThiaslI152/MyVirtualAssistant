# embedding.py

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma

CHROMA_DIR = "./chroma-data"

# Custom MPS-compatible HuggingFace embedder
class MpsHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
embeddings = MpsHuggingFaceEmbeddings(model=embedding_model)

# Set up Chroma vector store
vectorstore = Chroma(
    collection_name="langchain_collection",
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)
