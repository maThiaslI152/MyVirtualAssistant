from backend.services.processing.ingest import process_file
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any
import logging

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf",".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()
    result = process_file(file.filename, contents)

    return {"message": "File processed", "result": len(result)}