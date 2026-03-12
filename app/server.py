"""FastAPI translation server for Albanian ↔ English."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get(
    "TRANSLATOR_MODEL_DIR",
    "outputs/opusmt-alb-en-ft-3ep-full/final",
)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
RATE_LIMIT = os.environ.get("RATE_LIMIT", "30/minute")
MAX_INPUT_TOKENS = 512  # opus-mt models support up to 512 input tokens

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])

# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _tokenizer, _model
    model_path = Path(MODEL_DIR).resolve()
    if not model_path.exists():
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}. "
            "Set TRANSLATOR_MODEL_DIR or train a model first."
        )
    logger.info("Loading model from %s onto %s …", MODEL_DIR, DEVICE)
    _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    _model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    _model.to(DEVICE)
    _model.eval()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down — releasing model.")
    _model = None
    _tokenizer = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Albanian Translator",
    version="0.2.0",
    description=(
        "Fine-tuned Albanian ↔ English translation API. "
        "POST plain Albanian text to `/translate`, receive English output."
    ),
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Albanian text to translate")
    max_length: int = Field(512, ge=1, le=1024, description="Max output tokens (input is always truncated to 512 tokens)")
    num_beams: int = Field(4, ge=1, le=10, description="Beam search width")

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "Mirëdita, si jeni?", "max_length": 512, "num_beams": 4}]
        }
    }


class TranslateResponse(BaseModel):
    translation: str
    source_length: int
    model: str


class BatchTranslateRequest(BaseModel):
    texts: Annotated[list[str], Field(min_length=1, max_length=32)] = Field(
        ..., description="List of Albanian texts (max 32)"
    )
    max_length: int = Field(512, ge=1, le=1024)
    num_beams: int = Field(4, ge=1, le=10)


class BatchTranslateResponse(BaseModel):
    translations: list[str]
    count: int
    model: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", summary="Deep health check — verifies model inference")
def health():
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded")
    try:
        probe = _tokenizer("test", return_tensors="pt", truncation=True, max_length=16).to(DEVICE)
        with torch.no_grad():
            _model.generate(**probe, max_new_tokens=8)
    except Exception as exc:
        logger.error("Health probe failed: %s", exc)
        raise HTTPException(503, f"Model inference check failed: {exc}") from exc
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(DEVICE),
    }


@app.post(
    "/translate",
    response_model=TranslateResponse,
    summary="Translate a single Albanian text to English",
)
@limiter.limit(RATE_LIMIT)
def translate(req: TranslateRequest, request: Request):  # noqa: ARG001
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded yet")
    logger.debug("Translating %d chars", len(req.text))
    try:
        inputs = _tokenizer(
            req.text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=req.max_length,
                num_beams=req.num_beams,
            )
        translation = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError as exc:
        logger.error("GPU OOM during translation: %s", exc)
        raise HTTPException(503, "GPU out of memory — try a shorter input") from exc
    except Exception as exc:
        logger.exception("Translation failed")
        raise HTTPException(500, f"Translation failed: {exc}") from exc

    return TranslateResponse(
        translation=translation,
        source_length=len(req.text),
        model=Path(MODEL_DIR).name,
    )


@app.post(
    "/translate/batch",
    response_model=BatchTranslateResponse,
    summary="Translate multiple Albanian texts in a single request",
)
@limiter.limit("10/minute")
def translate_batch(req: BatchTranslateRequest, request: Request):  # noqa: ARG001
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded yet")
    logger.debug("Batch translating %d texts", len(req.texts))
    try:
        inputs = _tokenizer(
            req.texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=req.max_length,
                num_beams=req.num_beams,
            )
        translations = _tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError as exc:
        logger.error("GPU OOM during batch translation: %s", exc)
        raise HTTPException(503, "GPU out of memory — try fewer or shorter inputs") from exc
    except Exception as exc:
        logger.exception("Batch translation failed")
        raise HTTPException(500, f"Batch translation failed: {exc}") from exc

    return BatchTranslateResponse(
        translations=translations,
        count=len(translations),
        model=Path(MODEL_DIR).name,
    )


# ---------------------------------------------------------------------------
# Serve static UI
# ---------------------------------------------------------------------------
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(_static_dir / "index.html"))

    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
