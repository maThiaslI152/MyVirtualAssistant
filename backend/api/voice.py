from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
from io import BytesIO
from backend.services.processing.voice import VoiceProcessor

router = APIRouter()
voice_processor = VoiceProcessor()

class VoiceAnalysisResponse(BaseModel):
    text: str
    language: str
    confidence: float
    duration: float
    metadata: Dict[str, Any]

class SpeechGenerationRequest(BaseModel):
    text: str
    language: Optional[str] = "en"

@router.post("/transcribe", response_model=VoiceAnalysisResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio to text."""
    try:
        # Read audio data
        audio_data = await file.read()
        
        # Transcribe audio
        analysis = voice_processor.transcribe_audio(audio_data)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_speech(request: SpeechGenerationRequest):
    """Generate speech from text."""
    try:
        # Generate speech
        audio_data = voice_processor.generate_speech(
            request.text,
            request.language
        )
        
        # Convert to base64
        audio_base64 = base64.b64encode(audio_data).decode()
        
        return {"audio": audio_base64}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze")
async def analyze_voice(file: UploadFile = File(...)):
    """Analyze voice characteristics."""
    try:
        # Read audio data
        audio_data = await file.read()
        
        # Analyze voice
        analysis = voice_processor.analyze_voice(audio_data)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 