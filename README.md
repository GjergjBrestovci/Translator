# Albanian → English Training Starter (FineTranslations)

This project is scoped to Albanian-only subsets from `HuggingFaceFW/finetranslations`:
- `aln_Latn`
- `als_Latn`

## 1) Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Prepare Albanian-only dataset

This streams only the two Albanian subsets and writes train/validation/test JSONL files.

```bash
python scripts/prepare_dataset.py \
  --subsets aln_Latn als_Latn \
  --output-dir data/alb_en \
  --max-samples-per-subset 300000 \
  --drop-early-stop
```

Notes:
- Use `--max-samples-per-subset -1` to stream all rows (can take long).
- Output files: `data/alb_en/train.jsonl`, `validation.jsonl`, `test.jsonl`.

## 3) Fine-tune model

Default model is `Helsinki-NLP/opus-mt-sq-en`.

```bash
python scripts/train_albanian_to_english.py \
  --data-dir data/alb_en \
  --model-name Helsinki-NLP/opus-mt-sq-en \
  --output-dir outputs/opusmt-alb-en-ft \
  --num-train-epochs 2 \
  --fp16
```

## 4) Next improvements

- Increase `--max-samples-per-subset` gradually (start small, then scale up).
- Add a small human-evaluated dev set from your target domain.
- Try stronger models (NLLB/M2M100/T5 family) if you have more GPU memory.
