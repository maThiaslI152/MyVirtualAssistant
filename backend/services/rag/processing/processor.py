from typing import List, Dict, Any, Optional
import torch
import numpy as np
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer

class ContentProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Initialize device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {'MPS (Metal Performance Shaders) for GPU acceleration' if torch.backends.mps.is_available() else 'CPU'}")
        self.sentence_transformer = SentenceTransformer('intfloat/multilingual-e5-large', device=self.device)

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embeddings for the given text using the multilingual-e5-large model."""
        return self.sentence_transformer.encode([text])[0]

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            if end > text_length:
                end = text_length
            chunks.append(text[start:end])
            start = end - chunk_overlap

        return chunks

    def process_content(self, text: str, url: str = None) -> Dict[str, Any]:
        """Process text content and create chunks with embeddings."""
        try:
            # Create chunks
            chunks = self.chunk_text(text)
            
            # Process chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.get_embedding(chunk)
                
                # Create metadata
                metadata = {
                    'url': url,
                    'timestamp': datetime.utcnow().isoformat(),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.split())
                }

                processed_chunks.append({
                    'content': chunk,
                    'embedding': embedding,
                    'metadata': metadata
                })

            return {
                'chunks': processed_chunks,
                'total_chunks': len(chunks),
                'total_words': len(text.split())
            }
            
        except Exception as e:
            self.logger.error(f"Error processing content: {str(e)}")
            return None 