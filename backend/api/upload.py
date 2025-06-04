from fastapi import APIRouter, UploadFile, File, HTTPException
from services.ingest import process_file

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf",".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    contents = await file.read()
    result = process_file(file.filename, contents)

    return {"message": "File processed", "result": len(result)}