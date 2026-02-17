# Agent Reference — Albanian ↔ English Translator

> **Purpose**: This file is the single source of truth for any AI agent making
> changes to this project. Read it fully before editing any code.

---

## 1. Project Overview

Fine-tuned Albanian → English translation model based on
`Helsinki-NLP/opus-mt-sq-en` (MarianMT, ~300 MB seq2seq).

**Data source**: `HuggingFaceFW/finetranslations` — Albanian subsets:
- `aln_Latn` (Gheg Albanian, ~563 rows)
- `als_Latn` (Tosk Albanian, ~8.5 M rows, capped during download)

---

## 2. Directory Layout

```
Translator/
├── AGENT.md                 ← YOU ARE HERE — agent reference
├── README.md                ← user-facing docs
├── requirements.txt         ← Python deps (pip install -r)
├── .gitignore
├── data/
│   └── alb_en/
│       ├── train.jsonl      ← training split   (source/target pairs)
│       ├── validation.jsonl ← validation split
│       ├── test.jsonl       ← test split
│       └── metadata.json    ← split counts, subset info
├── scripts/
│   ├── prepare_dataset.py   ← streams HF data → local JSONL
│   └── train_albanian_to_english.py  ← fine-tunes MarianMT
├── app/
│   ├── server.py            ← FastAPI translation API
│   ├── translate.py         ← CLI inference tool
│   └── static/
│       └── index.html       ← browser UI (vanilla HTML/JS)
├── outputs/                 ← training checkpoints (git-ignored)
│   ├── opusmt-alb-en-ft/
│   ├── opusmt-alb-en-ft-rerun/
│   └── opusmt-alb-en-ft-3ep-full/
└── .venv/                   ← virtual env (git-ignored)
```

---

## 3. Data Format

Each JSONL row:
```json
{"source": "Albanian text…", "target": "English text…", "subset": "als_Latn", "id": "…"}
```

- `source` = Albanian original
- `target` = English translation
- Minimum 20 source chars, `early_stop=true` rows dropped

### Current dataset size
| Split      | Rows   |
|------------|--------|
| train      | 20,150 |
| validation | 205    |
| test       | 207    |

### Expanding the dataset
```bash
python scripts/prepare_dataset.py \
  --subsets aln_Latn als_Latn \
  --output-dir data/alb_en \
  --max-samples-per-subset 100000 \  # increase this
  --drop-early-stop
```
- Use `--max-samples-per-subset -1` for the full ~8.5 M als_Latn subset (very slow).
- Larger data is the **#1 lever** for improving quality.
- After expanding, retrain from scratch (do NOT continue from old checkpoint with new data).

---

## 4. Training

### Run training
```bash
source .venv/bin/activate
python scripts/train_albanian_to_english.py \
  --data-dir data/alb_en \
  --model-name Helsinki-NLP/opus-mt-sq-en \
  --output-dir outputs/<run-name> \
  --num-train-epochs 5.0 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --eval-steps 100 --save-steps 100 --logging-steps 20
```

### Key training facts
- Model: MarianMT (`AutoModelForSeq2SeqLM`)
- Tokenizer: SentencePiece via `AutoTokenizer`
- Metric: sacrebleu BLEU score (via `evaluate` library)
- Best model selected by: `metric_for_best_model="score"` (BLEU)
- Transformers v5: use `processing_class=tokenizer` (NOT `tokenizer=`)

### Quality targets
| Metric | Current (3ep / 5.5k) | Target |
|--------|---------------------|--------|
| BLEU   | 36.2                | ≥ 55   |
| chrF   | not tracked yet     | ≥ 70   |

> **"90% accuracy" in translation context**: There is no single "accuracy" number
> for translation. The proxy targets above (BLEU ≥ 55, chrF ≥ 70) represent
> **very high quality** machine translation. To reach them:
> 1. Scale training data to 100k–300k+ pairs
> 2. Train 3–5 epochs with early stopping
> 3. Consider switching to a larger model (NLLB-200, M2M-100) if MarianMT plateaus
> 4. Add chrF metric tracking alongside BLEU

---

## 5. Inference & UI

### CLI
```bash
python app/translate.py --model-dir outputs/<best-run>/final "Përshëndetje, si jeni?"
```

### API server
```bash
uvicorn app.server:app --host 0.0.0.0 --port 8000
# POST /translate  {"text": "Albanian text", "direction": "sq-en"}
# GET  /health
```

### Web UI
Open `http://localhost:8000` after starting the server.
The UI is served from `app/static/index.html`.

---

## 6. Rules for Agents

1. **Always read this file first** before making changes.
2. **Never modify training data files directly** — use `prepare_dataset.py`.
3. **Never hard-code model paths** — use CLI args or env vars.
4. **Keep `compute_metrics` safe**: sacrebleu returns dicts with list values;
   always guard `round()` with `isinstance(value, (int, float))`.
5. **Transformers v5 API**: use `processing_class=` not `tokenizer=` in Trainer.
6. **Test after every change**: run `python -c "from app.server import app"` to
   verify imports, and run a quick training sanity check with `--num-train-epochs 0.01`.
7. **Git-ignore large files**: `outputs/`, `data/`, `.venv/` must stay in `.gitignore`.
8. **requirements.txt**: keep pinned to minimum versions, add new deps there.
9. **JSONL format**: do not change the schema (`source`, `target`, `subset`, `id`).
10. **UI changes**: keep the UI self-contained in `app/static/index.html` (no build step).

---

## 7. Roadmap

- [ ] Expand dataset to 100k+ pairs
- [ ] Train to BLEU ≥ 55 / chrF ≥ 70
- [ ] Add chrF metric to training script
- [ ] Add English → Albanian direction
- [ ] Add language auto-detection
- [ ] Add batch translation endpoint
- [ ] Add conversation/chat mode in UI
- [ ] Try NLLB-200 or M2M-100 if MarianMT plateaus
- [ ] Deploy behind nginx/gunicorn for production
