from fastapi import APIRouter
from api.upload import router as upload_router
from api.chat import router as chat_router

router = APIRouter(prefix="/api")
router.include_router(upload_router)
router.include_router(chat_router)

@router.get("/ping")
async def root():
    return {"message": "Pong"}