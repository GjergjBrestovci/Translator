"""FastAPI translation server for Albanian ↔ English."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get(
    "TRANSLATOR_MODEL_DIR",
    "outputs/opusmt-alb-en-ft-3ep-full/final",
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Albanian Translator", version="0.1.0")

# Global model references (loaded on startup)
_tokenizer = None
_model = None


@app.on_event("startup")
def _load_model() -> None:
    global _tokenizer, _model
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}. "
            "Set TRANSLATOR_MODEL_DIR or train a model first."
        )
    _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    _model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    _model.eval()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to translate")
    max_length: int = Field(512, ge=1, le=1024)
    num_beams: int = Field(4, ge=1, le=10)


class TranslateResponse(BaseModel):
    translation: str
    source_length: int
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_dir": MODEL_DIR}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded yet")

    inputs = _tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=req.max_length,
    )
    outputs = _model.generate(
        **inputs,
        max_new_tokens=req.max_length,
        num_beams=req.num_beams,
    )
    translation = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    return TranslateResponse(
        translation=translation,
        source_length=len(req.text),
        model=MODEL_DIR,
    )


# ---------------------------------------------------------------------------
# Serve static UI
# ---------------------------------------------------------------------------
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    @app.get("/")
    def index():
        return FileResponse(str(_static_dir / "index.html"))

    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
