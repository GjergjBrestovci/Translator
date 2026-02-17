# Albanian → English Translator

Fine-tuned MarianMT model for Albanian → English translation, with a
web UI and REST API for interactive use.

> **Agent note**: See [AGENT.md](AGENT.md) for the full technical reference
> before making any changes to this project.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
```

## 1) Prepare Dataset

```bash
python scripts/prepare_dataset.py \
  --subsets aln_Latn als_Latn \
  --output-dir data/alb_en \
  --max-samples-per-subset 100000 \
  --drop-early-stop
```

## 2) Train

```bash
python scripts/train_albanian_to_english.py \
  --data-dir data/alb_en \
  --model-name Helsinki-NLP/opus-mt-sq-en \
  --output-dir outputs/opusmt-alb-en-ft \
  --num-train-epochs 5 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --eval-steps 100 --save-steps 100 --logging-steps 20
```

## 3) Translate (CLI)

```bash
python app/translate.py --model-dir outputs/<run>/final "Përshëndetje, si jeni?"
# or interactive mode (omit text argument)
python app/translate.py --model-dir outputs/<run>/final
```

## 4) Web UI & API

```bash
# Set model path (or use default)
export TRANSLATOR_MODEL_DIR=outputs/opusmt-alb-en-ft-3ep-full/final

uvicorn app.server:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000 in your browser
```

### API Endpoints

| Method | Path         | Description               |
|--------|--------------|---------------------------|
| GET    | `/health`    | Model status              |
| POST   | `/translate` | Translate Albanian → English |
| GET    | `/`          | Web UI                    |

### POST /translate

```json
{"text": "Mirëdita, si jeni?", "max_length": 512, "num_beams": 4}
```

Response:
```json
{"translation": "Good day, how are you?", "source_length": 19, "model": "..."}
```

## Roadmap

See [AGENT.md](AGENT.md) § Roadmap for full details.
