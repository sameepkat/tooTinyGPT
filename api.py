import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import get_inference_engine


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.8, gt=0)
    top_k: int | None = Field(default=50, ge=1)


class GenerateResponse(BaseModel):
    prompt: str
    response: str


app = FastAPI(title="tooTinyGPT API", version="0.0.1")


def _engine_from_environment():
    checkpoint_path = os.environ.get("TOOTINYGPT_CHECKPOINT")
    tokenizer_path = os.environ.get("TOOTINYGPT_TOKENIZER")
    device = os.environ.get("TOOTINYGPT_DEVICE", "cpu")
    if not checkpoint_path or not tokenizer_path:
        raise RuntimeError("TOOTINYGPT_CHECKPOINT and TOOTINYGPT_TOKENIZER must be configured")
    return get_inference_engine(checkpoint_path, tokenizer_path, device)


@app.get("/")
def home():
    return {"message": "tooTinyGPT API is running"}


@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    try:
        response = _engine_from_environment().generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Generation failed: {error}") from error
    return GenerateResponse(prompt=request.prompt, response=response)
