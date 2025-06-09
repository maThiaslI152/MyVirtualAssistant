from typing import List, Dict, Any, Optional
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import trafilatura
from newspaper import Article
import logging
from datetime import datetime
import re

class WebSearchService:
    def __init__(
        self,
        max_depth: int = 2,
        max_pages: int = 10,
        timeout: int = 30,
        allowed_domains: Optional[List[str]] = None
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.timeout = timeout
        self.allowed_domains = allowed_domains or []
        self.visited_urls = set()
        self.session = None
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    async def close(self):
        """Close aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None

    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and allowed."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            if self.allowed_domains and parsed.netloc not in self.allowed_domains:
                return False
            return True
        except Exception:
            return False

    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                return None
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            full_url = urljoin(base_url, href)
            if self.is_valid_url(full_url):
                links.append(full_url)
        return links

    def extract_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract and process content from HTML."""
        try:
            # Try trafilatura first (better for article content)
            content = trafilatura.extract(html)
            if not content:
                # Fallback to newspaper3k
                article = Article(url)
                article.set_html(html)
                article.parse()
                content = article.text

            # Clean and process content
            content = re.sub(r'\s+', ' ', content).strip()
            
            return {
                'content': content,
                'url': url,
                'timestamp': datetime.utcnow().isoformat(),
                'word_count': len(content.split())
            }
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    async def crawl(self, start_url: str) -> List[Dict[str, Any]]:
        """Crawl web pages starting from a URL."""
        await self.initialize()
        results = []
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        
        try:
            while urls_to_visit and len(results) < self.max_pages:
                url, depth = urls_to_visit.pop(0)
                
                if url in self.visited_urls or depth > self.max_depth:
                    continue
                
                self.visited_urls.add(url)
                html = await self.fetch_page(url)
                
                if html:
                    # Extract and process content
                    content_data = self.extract_content(html, url)
                    if content_data and content_data['content']:
                        results.append(content_data)
                    
                    # Extract links for next level
                    if depth < self.max_depth:
                        links = self.extract_links(html, url)
                        urls_to_visit.extend([(link, depth + 1) for link in links])
                
                # Small delay to be nice to servers
                await asyncio.sleep(0.5)
                
        finally:
            await self.close()
            
        return results

    async def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web and return processed results."""
        # This is a placeholder for actual search implementation
        # You would typically integrate with a search API here
        # For now, we'll just crawl a few pages
        results = await self.crawl(f"https://example.com/search?q={query}")
        return results[:num_results] 