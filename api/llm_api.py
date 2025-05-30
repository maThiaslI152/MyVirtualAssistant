from fastapi import FastAPI
from llama_cpp import Llama
from config import settings

llm = Llama(
    model_path=settings.MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=settings.MAX_TOKENS,
    use_mlock=True,
    use_mmap=True,
    verbose=settings.DEBUG
)

app = FastAPI()

@app.get("/generate")
def generate(prompt: str):
    output = llm(
        prompt,
        max_tokens=settings.MAX_TOKENS,
        temperature=settings.TEMPERATURE,
        top_p=settings.TOP_P,
        frequency_penalty=settings.FREQUENCY_PENALTY,
        presence_penalty=settings.PRESENCE_PENALTY
    )
    try:
        return {"response": output["choices"][0]["text"].strip()}
    except (KeyError, IndexError):
        return {"error": "Model failed to generate a response."}

@app.get("/api_health")
def api_health_check():
    return {"status": "ok"}
