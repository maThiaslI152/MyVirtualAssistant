from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, UploadFile, File
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
from services.web_search_service import WebSearchService
from services.rag_search_service import RAGSearchService
from services.cache_service import CacheService
from services.content_processor import ContentProcessor
from services.search_history_service import SearchHistoryService
from vectorstore.db import get_vector_store
from config import settings
from services.file_processor_service import FileProcessorService
import logging
import os
from pathlib import Path
import shutil
import uuid

router = APIRouter(prefix=settings.API_V1_STR + "/rag")
logger = logging.getLogger(__name__)

# Define the base path for file processing
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

class RAGSearchRequest(BaseModel):
    query: str
    num_web_results: int = 5
    num_retrieval_results: int = 3
    allowed_domains: Optional[List[str]] = None
    extract_summary: bool = True
    extract_entities: bool = True
    extract_keywords: bool = True
    cache_ttl: Optional[int] = None
    file_types: Optional[List[str]] = None
    user_id: Optional[str] = None
    get_recommendations: bool = False
    summary_method: str = 'hybrid'
    summary_min_length: int = 50
    summary_max_length: int = 150
    summary_compression_ratio: float = 0.3

class RAGSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Optional[Dict[str, Any]] = None
    entities: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    search_history: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    summary_metadata: Optional[Dict[str, Any]] = None

class FileProcessResponse(BaseModel):
    status: str
    file_path: str
    file_type: str
    chunks: int
    metadata: Dict[str, Any]

class FileSearchRequest(BaseModel):
    query: str
    file_types: Optional[List[str]] = None
    num_results: int = 5

class FileSearchResponse(BaseModel):
    status: str
    query: str
    results: List[Dict[str, Any]]
    total: int

class SearchHistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int

class SearchStatsResponse(BaseModel):
    total_searches: int
    total_urls: int
    total_domains: int
    recent_searches: List[Dict[str, Any]]
    domain_distribution: Dict[str, int]

class SearchAnalyticsResponse(BaseModel):
    time_analysis: Dict[str, Any]
    query_analysis: Dict[str, Any]
    url_analysis: Dict[str, Any]
    network_analysis: Dict[str, Any]

class URLInsightsResponse(BaseModel):
    url: str
    domain: str
    first_seen: Optional[str]
    last_seen: Optional[str]
    search_count: int
    related_urls: List[Dict[str, Any]]

# Initialize services
web_search_service = WebSearchService()
vector_store = get_vector_store()
cache_service = CacheService()
content_processor = ContentProcessor()
search_history_service = SearchHistoryService()
file_processor = FileProcessorService(base_path=BASE_PATH)
rag_service = RAGSearchService(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    cache_service=cache_service,
    content_processor=content_processor,
    file_processor=file_processor,
    web_search=web_search_service
)

@router.post("/search", response_model=RAGSearchResponse)
async def rag_search(request: RAGSearchRequest, background_tasks: BackgroundTasks):
    """Perform a RAG search with history tracking."""
    try:
        # Get search results
        results = await rag_service.rag_search(
            query=request.query,
            num_web_results=request.num_web_results,
            num_retrieval_results=request.num_retrieval_results,
            allowed_domains=request.allowed_domains,
            extract_summary=request.extract_summary,
            extract_entities=request.extract_entities,
            extract_keywords=request.extract_keywords,
            cache_ttl=request.cache_ttl,
            file_types=request.file_types,
            summary_method=request.summary_method,
            summary_min_length=request.summary_min_length,
            summary_max_length=request.summary_max_length,
            summary_compression_ratio=request.summary_compression_ratio
        )
        
        # Get search history if user_id is provided
        search_history = None
        if request.user_id:
            search_history = await search_history_service.get_search_history(request.user_id)
            background_tasks.add_task(
                search_history_service.add_search,
                request.user_id,
                request.query,
                results
            )
        
        # Get recommendations if requested
        recommendations = None
        if request.get_recommendations and request.user_id:
            recommendations = await search_history_service.get_recommendations(request.user_id)
        
        return {
            'results': results['web_results'],
            'summary': results.get('summary'),
            'entities': results.get('entities'),
            'keywords': results.get('keywords'),
            'search_history': search_history,
            'recommendations': recommendations,
            'summary_metadata': results.get('summary_metadata')
        }
    except Exception as e:
        logger.error(f"Error in RAG search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-file", response_model=FileProcessResponse)
async def process_file(file: UploadFile = File(...)):
    """Process a file and add it to the vector store."""
    try:
        # Create upload directory if it doesn't exist
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process file
        result = await rag_service.process_file(str(file_path))
        return result
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process-files")
async def process_files(files: List[UploadFile] = File(...)):
    """Process multiple files concurrently."""
    try:
        # Create upload directory if it doesn't exist
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded files
        file_paths = []
        for file in files:
            file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append(str(file_path))

        # Process files
        results = await rag_service.process_files(file_paths)
        return results
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search-files", response_model=FileSearchResponse)
async def search_files(request: FileSearchRequest):
    """Search through processed files."""
    try:
        results = await rag_service.search_files(
            query=request.query,
            file_types=request.file_types,
            num_results=request.num_results
        )
        return results
    except Exception as e:
        logger.error(f"Error searching files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-file-types")
async def get_supported_file_types():
    """Get list of supported file types."""
    return {
        "code_files": file_processor.supported_code_extensions,
        "document_files": [".pdf"],
        "image_files": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"]
    }

@router.get("/history/{user_id}", response_model=SearchHistoryResponse)
async def get_search_history(
    user_id: str,
    page: int = 1,
    page_size: int = 10
):
    """Get user's search history with pagination."""
    try:
        offset = (page - 1) * page_size
        history = search_history_service.get_search_history(
            user_id=user_id,
            limit=page_size,
            offset=offset
        )
        total = len(search_history_service.get_search_history(user_id))
        
        return {
            'history': history,
            'total': total,
            'page': page,
            'page_size': page_size
        }
    except Exception as e:
        logger.error(f"Error getting search history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/{user_id}", response_model=SearchStatsResponse)
async def get_search_stats(user_id: str):
    """Get search statistics for a user."""
    try:
        stats = search_history_service.get_search_stats(user_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting search stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/{user_id}", response_model=SearchAnalyticsResponse)
async def get_search_analytics(user_id: str):
    """Get advanced analytics about user's search behavior."""
    try:
        analytics = search_history_service.get_analytics(user_id)
        return analytics
    except Exception as e:
        logger.error(f"Error getting search analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/{user_id}")
async def get_recommendations(user_id: str, query: str):
    """Get search recommendations based on user history."""
    try:
        recommendations = search_history_service.get_recommendations(
            user_id=user_id,
            query=query
        )
        return recommendations
    except Exception as e:
        logger.error(f"Error getting search recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/url-insights/{user_id}", response_model=URLInsightsResponse)
async def get_url_insights(user_id: str, url: str):
    """Get detailed insights about a specific URL."""
    try:
        insights = search_history_service.get_url_insights(user_id, url)
        return insights
    except Exception as e:
        logger.error(f"Error getting URL insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/history/{user_id}")
async def clear_history(user_id: str):
    """Clear user's search history."""
    try:
        success = search_history_service.clear_history(user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear history")
        return {"message": "History cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing search history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/urls/{user_id}")
async def get_searched_urls(user_id: str):
    """Get list of URLs searched by user."""
    try:
        urls = search_history_service.get_searched_urls(user_id)
        return {"urls": list(urls)}
    except Exception as e:
        logger.error(f"Error getting searched URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/check-url/{user_id}")
async def check_url(user_id: str, url: str):
    """Check if a URL has been searched by user."""
    try:
        is_searched = search_history_service.is_url_searched(user_id, url)
        return {"is_searched": is_searched}
    except Exception as e:
        logger.error(f"Error checking searched URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    await web_search_service.initialize()
    await cache_service.initialize()
    try:
        # Create necessary directories
        Path("./data/chroma").mkdir(parents=True, exist_ok=True)
        Path("./data/uploads").mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@router.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    await web_search_service.close()
    await cache_service.close()
    try:
        # Clean up temporary files
        upload_dir = Path("./data/uploads")
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        raise 