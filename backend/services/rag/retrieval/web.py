import asyncio
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from playwright.async_api import async_playwright
import logging
import uuid
from datetime import datetime

from ..core.rag_config import VECTORSTORE_CONFIG

logger = logging.getLogger(__name__)

class WebSearchService:
    def __init__(self):
        self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
        self.vector_store = Chroma(
            collection_name=VECTORSTORE_CONFIG["collection_name"],
            embedding_function=self.embedding_model.encode,
            persist_directory=VECTORSTORE_CONFIG["persist_directory"]
        )

    async def fetch_page(self, url: str) -> str:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=20000)
                content = await page.content()
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                content = ""
            await browser.close()
        return content

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Remove scripts and styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text

    async def process_and_store(self, url: str) -> Optional[Dict[str, Any]]:
        html = await self.fetch_page(url)
        if not html:
            return None
        text = self.extract_text(html)
        if not text or len(text) < 100:
            logger.warning(f"Not enough text extracted from {url}")
            return None
        embedding = self.embedding_model.encode(text)
        metadata = {
            "url": url,
            "timestamp": datetime.utcnow().isoformat(),
            "id": str(uuid.uuid4()),
            "length": len(text)
        }
        # Store in Chroma
        self.vector_store.add_texts([text], metadatas=[metadata])
        self.vector_store.persist()
        return {"url": url, "text": text, "embedding": embedding, "metadata": metadata}

    async def search_and_store(self, urls: List[str]) -> List[Dict[str, Any]]:
        results = []
        for url in urls:
            result = await self.process_and_store(url)
            if result:
                results.append(result)
        return results 