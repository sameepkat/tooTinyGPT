from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import inference
import random


# -----------------------------
# Request and response formats
# -----------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    encoding: str = "gpt2"


class GenerateResponse(BaseModel):
    prompt: str
    response: str


# -----------------------------
# FastAPI application
# -----------------------------

app = FastAPI(
    title="tooTinyGPT API",
    version="0.0.1",
)



@app.get("/")
def home():
    return {"message": "tooTinyGPT API is running"}


@app.post("/generate", response_model=GenerateResponse)
def generate_text(
    request: GenerateRequest,
):
    try:
        generated_text = inference.infer(
            prompt=request.prompt,
            checkpoint_file=Path("checkpoint.pt"),
            encoding=request.encoding,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {error}",
        )


    return GenerateResponse(prompt=request.prompt, response=generated_text)



