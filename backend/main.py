import asyncio
from fastapi import FastAPI
from api.routes import router
from vectorstore.memory import redis_expiry_listener

#setup FastAPI
app = FastAPI(title="Owlynn AI Assistant")

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Owlynn backend."}

@app.on_event("startup")
async def start_listener():
    asyncio.create_task(redis_expiry_listener())
