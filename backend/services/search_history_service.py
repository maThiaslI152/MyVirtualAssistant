from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
import json
import logging
from redis import Redis
from redis.exceptions import RedisError
import hashlib
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from urllib.parse import urlparse, parse_qs
import re
from concurrent.futures import ThreadPoolExecutor
import asyncio
from config import settings

logger = logging.getLogger(__name__)

class SearchHistoryService:
    def __init__(self):
        self.redis_client = self._init_redis()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.search_patterns = defaultdict(list)
        self.url_graph = nx.DiGraph()
        
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            client = Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            return client
        except RedisError as e:
            logger.error(f"Failed to initialize Redis: {e}")
            return None

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing common variations."""
        try:
            parsed = urlparse(url)
            # Remove www. prefix
            netloc = parsed.netloc.replace('www.', '')
            # Remove trailing slashes
            path = parsed.path.rstrip('/')
            # Remove common tracking parameters
            query = parse_qs(parsed.query)
            filtered_query = {k: v for k, v in query.items() 
                            if k not in ['utm_source', 'utm_medium', 'utm_campaign']}
            # Reconstruct URL
            return f"{parsed.scheme}://{netloc}{path}"
        except Exception as e:
            logger.error(f"Error normalizing URL {url}: {e}")
            return url

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            return urlparse(url).netloc.replace('www.', '')
        except Exception as e:
            logger.error(f"Error extracting domain from {url}: {e}")
            return url

    def add_search(self, user_id: str, query: str, results: List[Dict[str, Any]]) -> str:
        """Add a search to user's history with enhanced tracking."""
        try:
            search_id = hashlib.md5(f"{user_id}:{query}:{datetime.now().isoformat()}".encode()).hexdigest()
            timestamp = datetime.now().isoformat()
            
            # Store search record
            search_record = {
                'id': search_id,
                'query': query,
                'timestamp': timestamp,
                'result_count': len(results)
            }
            
            # Store in Redis with TTL
            self.redis_client.hset(
                f"search_history:{user_id}",
                search_id,
                json.dumps(search_record)
            )
            self.redis_client.expire(f"search_history:{user_id}", settings.SEARCH_HISTORY_TTL)
            
            # Track URLs and domains
            for result in results:
                url = self._normalize_url(result['url'])
                domain = self._extract_domain(url)
                
                # Add to URL set
                self.redis_client.sadd(f"searched_urls:{user_id}", url)
                self.redis_client.expire(f"searched_urls:{user_id}", settings.SEARCH_HISTORY_TTL)
                
                # Add to domain set
                self.redis_client.sadd(f"searched_domains:{user_id}", domain)
                self.redis_client.expire(f"searched_domains:{user_id}", settings.SEARCH_HISTORY_TTL)
                
                # Update URL graph
                self.url_graph.add_edge(query, url, weight=1)
                self.url_graph.add_edge(domain, url, weight=1)
            
            # Update search patterns
            self._update_search_patterns(user_id, query, results)
            
            return search_id
        except Exception as e:
            logger.error(f"Error adding search to history: {e}")
            return None

    def _update_search_patterns(self, user_id: str, query: str, results: List[Dict[str, Any]]):
        """Update search patterns based on query and results."""
        try:
            # Extract keywords from query
            keywords = set(re.findall(r'\w+', query.lower()))
            
            # Update pattern frequency
            for keyword in keywords:
                self.search_patterns[user_id].append({
                    'keyword': keyword,
                    'timestamp': datetime.now().isoformat(),
                    'result_count': len(results)
                })
            
            # Keep only recent patterns
            self.search_patterns[user_id] = self.search_patterns[user_id][-settings.MAX_HISTORY_PER_USER:]
        except Exception as e:
            logger.error(f"Error updating search patterns: {e}")

    def get_search_history(self, user_id: str, limit: int = 10, offset: int = 0) -> List[Dict[str, Any]]:
        """Get user's search history with pagination."""
        try:
            history = []
            search_records = self.redis_client.hgetall(f"search_history:{user_id}")
            
            # Sort by timestamp
            sorted_records = sorted(
                search_records.items(),
                key=lambda x: json.loads(x[1])['timestamp'],
                reverse=True
            )
            
            # Apply pagination
            paginated_records = sorted_records[offset:offset + limit]
            
            for search_id, record in paginated_records:
                history.append(json.loads(record))
            
            return history
        except Exception as e:
            logger.error(f"Error getting search history: {e}")
            return []

    def get_searched_urls(self, user_id: str) -> Set[str]:
        """Get list of URLs searched by user."""
        try:
            return self.redis_client.smembers(f"searched_urls:{user_id}")
        except Exception as e:
            logger.error(f"Error getting searched URLs: {e}")
            return set()

    def is_url_searched(self, user_id: str, url: str) -> bool:
        """Check if URL has been searched by user."""
        try:
            normalized_url = self._normalize_url(url)
            return self.redis_client.sismember(f"searched_urls:{user_id}", normalized_url)
        except Exception as e:
            logger.error(f"Error checking URL: {e}")
            return False

    def get_search_stats(self, user_id: str) -> Dict[str, Any]:
        """Get enhanced search statistics."""
        try:
            # Get basic stats
            total_searches = len(self.redis_client.hgetall(f"search_history:{user_id}"))
            total_urls = len(self.redis_client.smembers(f"searched_urls:{user_id}"))
            total_domains = len(self.redis_client.smembers(f"searched_domains:{user_id}"))
            
            # Get recent searches
            recent_searches = self.get_search_history(user_id, limit=5)
            
            # Get domain distribution
            domains = self.redis_client.smembers(f"searched_domains:{user_id}")
            domain_counts = Counter(domains)
            
            return {
                'total_searches': total_searches,
                'total_urls': total_urls,
                'total_domains': total_domains,
                'recent_searches': recent_searches,
                'domain_distribution': dict(domain_counts),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting search stats: {e}")
            return {}

    def get_recommendations(self, user_id: str, query: str) -> List[Dict[str, Any]]:
        """Get search recommendations based on user history."""
        try:
            # Get user's search history
            history = self.get_search_history(user_id, limit=settings.MAX_HISTORY_PER_USER)
            if not history:
                return []
            
            # Prepare data for similarity analysis
            queries = [h['query'] for h in history]
            queries.append(query)
            
            # Compute TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform(queries)
            
            # Compute similarity scores
            similarity_scores = cosine_similarity(
                tfidf_matrix[-1:],  # Current query
                tfidf_matrix[:-1]   # Historical queries
            )[0]
            
            # Get recommendations
            recommendations = []
            for idx, score in enumerate(similarity_scores):
                if score >= settings.RECOMMENDATION_THRESHOLD:
                    recommendations.append({
                        'query': queries[idx],
                        'similarity': float(score),
                        'timestamp': history[idx]['timestamp'],
                        'result_count': history[idx]['result_count']
                    })
            
            # Sort by similarity and limit
            recommendations.sort(key=lambda x: x['similarity'], reverse=True)
            return recommendations[:settings.MAX_RECOMMENDATIONS]
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []

    def get_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get advanced analytics about user's search behavior."""
        try:
            history = self.get_search_history(user_id, limit=settings.MAX_HISTORY_PER_USER)
            if not history:
                return {}
            
            # Time-based analysis
            timestamps = [datetime.fromisoformat(h['timestamp']) for h in history]
            hourly_distribution = Counter(t.hour for t in timestamps)
            daily_distribution = Counter(t.date() for t in timestamps)
            
            # Query analysis
            queries = [h['query'] for h in history]
            query_lengths = [len(q.split()) for q in queries]
            avg_query_length = sum(query_lengths) / len(query_lengths)
            
            # URL analysis
            urls = self.get_searched_urls(user_id)
            domains = [self._extract_domain(url) for url in urls]
            domain_distribution = Counter(domains)
            
            # Network analysis
            url_centrality = nx.pagerank(self.url_graph)
            
            return {
                'time_analysis': {
                    'hourly_distribution': dict(hourly_distribution),
                    'daily_distribution': {str(k): v for k, v in daily_distribution.items()}
                },
                'query_analysis': {
                    'total_queries': len(queries),
                    'avg_query_length': avg_query_length,
                    'unique_queries': len(set(queries))
                },
                'url_analysis': {
                    'total_urls': len(urls),
                    'total_domains': len(set(domains)),
                    'domain_distribution': dict(domain_distribution)
                },
                'network_analysis': {
                    'central_urls': dict(sorted(url_centrality.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True)[:10])
                },
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {}

    def get_url_insights(self, user_id: str, url: str) -> Dict[str, Any]:
        """Get detailed insights about a specific URL."""
        try:
            normalized_url = self._normalize_url(url)
            domain = self._extract_domain(url)
            
            # Get URL metadata
            first_seen = None
            last_seen = None
            search_count = 0
            
            history = self.get_search_history(user_id)
            for record in history:
                if normalized_url in record.get('urls', []):
                    if not first_seen:
                        first_seen = record['timestamp']
                    last_seen = record['timestamp']
                    search_count += 1
            
            # Get related URLs
            related_urls = []
            if url in self.url_graph:
                for neighbor in self.url_graph.neighbors(url):
                    if neighbor != url:
                        related_urls.append({
                            'url': neighbor,
                            'weight': self.url_graph[url][neighbor]['weight']
                        })
            
            return {
                'url': normalized_url,
                'domain': domain,
                'first_seen': first_seen,
                'last_seen': last_seen,
                'search_count': search_count,
                'related_urls': sorted(related_urls, 
                                     key=lambda x: x['weight'], 
                                     reverse=True)[:5],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting URL insights: {e}")
            return {}

    def clear_history(self, user_id: str) -> bool:
        """Clear user's search history."""
        try:
            # Delete all related keys
            keys = [
                f"search_history:{user_id}",
                f"searched_urls:{user_id}",
                f"searched_domains:{user_id}"
            ]
            self.redis_client.delete(*keys)
            
            # Clear in-memory data
            if user_id in self.search_patterns:
                del self.search_patterns[user_id]
            
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False 