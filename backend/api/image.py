from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import base64
from io import BytesIO
from ..services.image_processor import ImageProcessor

router = APIRouter()
image_processor = ImageProcessor()

class ImageAnalysisResponse(BaseModel):
    objects: List[Dict[str, Any]]
    faces: List[Dict[str, Any]]
    text: List[Dict[str, Any]]
    colors: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""

class ImageEditRequest(BaseModel):
    prompt: str

@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze image and extract features."""
    try:
        # Read image data
        image_data = await file.read()
        
        # Analyze image
        analysis = image_processor.analyze_image(image_data)
        
        return analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    """Generate image from text prompt."""
    try:
        # Generate image
        image_data = image_processor.generate_image(
            request.prompt,
            request.negative_prompt
        )
        
        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode()
        
        return {"image": image_base64}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/edit")
async def edit_image(
    file: UploadFile = File(...),
    request: ImageEditRequest = None
):
    """Edit image based on text prompt."""
    try:
        # Read image data
        image_data = await file.read()
        
        # Edit image
        edited_image = image_processor.edit_image(image_data, request.prompt)
        
        # Convert to base64
        image_base64 = base64.b64encode(edited_image).decode()
        
        return {"image": image_base64}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/remove-background")
async def remove_background(file: UploadFile = File(...)):
    """Remove background from image."""
    try:
        # Read image data
        image_data = await file.read()
        
        # Remove background
        output = image_processor.remove_background(image_data)
        
        # Convert to base64
        image_base64 = base64.b64encode(output).decode()
        
        return {"image": image_base64}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 