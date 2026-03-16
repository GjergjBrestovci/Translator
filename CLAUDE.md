# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuned Albanian → English translation system built on `Helsinki-NLP/opus-mt-sq-en` (MarianMT, ~300 MB seq2seq). See `AGENT.md` for the full agent reference including data format, training details, and roadmap.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # adds pytest, httpx
```

## Common Commands

**Lint:**
```bash
ruff check app/ scripts/
```

**Tests** (uses mocked models — no checkpoint required):
```bash
pytest tests/ -v --tb=short
```

**Verify imports after changes:**
```bash
python -c "from app.server import app"
```

**Training sanity check (fast):**
```bash
python scripts/train_albanian_to_english.py \
  --data-dir data/alb_en \
  --model-name Helsinki-NLP/opus-mt-sq-en \
  --output-dir outputs/test-run \
  --num-train-epochs 0.01
```

**Start API server:**
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

**Docker:**
```bash
docker build -t translator .
docker run -v /path/to/model:/models -e TRANSLATOR_MODEL_DIR=/models -p 8000:8000 translator
```

**CLI translation:**
```bash
python app/translate.py --model-dir outputs/<run>/final "Albanian text here"
```

## Architecture

```
scripts/prepare_dataset.py   → streams HuggingFace data → local JSONL in data/
scripts/train_albanian_to_english.py  → fine-tunes MarianMT, saves to outputs/<run>/final
app/server.py                → FastAPI server, loads model at startup via lifespan()
app/translate.py             → CLI wrapper (single text or interactive mode)
app/static/index.html        → self-contained browser UI (no build step)
```

**Data flow**: `prepare_dataset.py` → `data/alb_en/{train,validation,test}.jsonl` → `train_albanian_to_english.py` → `outputs/<run>/final/` → `server.py` / `translate.py`

**Model config**: `TRANSLATOR_MODEL_DIR` env var (or `--model-dir` CLI arg) controls which checkpoint is loaded. Default: `outputs/opusmt-alb-en-ft-3ep-full/final`.

**Rate limiting**: `RATE_LIMIT` env var (default `30/minute` for `/translate`, `10/minute` for `/translate/batch`). `ALLOWED_ORIGINS` controls CORS.

## Key Constraints

- **Transformers v5 API**: use `processing_class=tokenizer` (not `tokenizer=`) in `Seq2SeqTrainer`.
- **`compute_metrics`**: sacrebleu returns dicts with list values — always guard `round()` with `isinstance(value, (int, float))`.
- Never modify JSONL files directly; use `prepare_dataset.py`. JSONL schema: `source`, `target`, `subset`, `id`.
- `outputs/`, `data/`, `.venv/` are git-ignored — never commit them.
- UI must remain self-contained in `app/static/index.html` with no build step.
