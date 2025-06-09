from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from backend.services.processing.language import LanguageProcessor

router = APIRouter()
language_processor = LanguageProcessor()

class TextRequest(BaseModel):
    text: str
    target_language: Optional[str] = None

class LanguageAnalysisResponse(BaseModel):
    language: str
    confidence: float
    tokens: List[str]
    entities: List[Dict[str, Any]]
    sentiment: Dict[str, float]

@router.post("/detect")
async def detect_language(request: TextRequest):
    """Detect language of text."""
    try:
        # Detect language
        result = language_processor.detect_language(request.text)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate")
async def translate_text(request: TextRequest):
    """Translate text to target language."""
    try:
        if not request.target_language:
            raise HTTPException(
                status_code=400,
                detail="target_language is required"
            )
        
        # Translate text
        translated_text = language_processor.translate_text(
            request.text,
            request.target_language
        )
        
        return {"translated_text": translated_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", response_model=LanguageAnalysisResponse)
async def analyze_text(request: TextRequest):
    """Analyze text in any language."""
    try:
        # Analyze text
        analysis = language_processor.analyze_text(request.text)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported")
async def get_supported_languages():
    """Get list of supported languages."""
    try:
        # Get supported languages
        languages = language_processor.get_supported_languages()
        
        return {"languages": languages}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 