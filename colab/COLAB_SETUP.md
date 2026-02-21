# Colab Setup (Translator Project)

Use these cells in a new Colab notebook to train this project with GPU and save progress to Drive.

## 1) Runtime + Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then set runtime to **GPU**:
- Runtime → Change runtime type → Hardware accelerator: GPU

## 2) Clone/Sync Project Into Drive

Choose one:

### Option A — GitHub clone (recommended)
```bash
%cd /content/drive/MyDrive
!git clone https://github.com/<your-user>/<your-repo>.git Translator || true
%cd /content/drive/MyDrive/Translator
!git pull
```

### Option B — Upload zipped project manually
Upload your project zip to Drive, then:
```bash
%cd /content/drive/MyDrive
!unzip -o Translator.zip -d Translator
%cd /content/drive/MyDrive/Translator
```

## 3) Install Dependencies

```bash
%cd /content/drive/MyDrive/Translator
!python -m pip install -U pip
!pip install -r requirements.txt
```

## 4) (Optional) Rebuild/Expand Dataset in Colab

This uses the stable rows API path you already added.

```bash
%cd /content/drive/MyDrive/Translator
!PYTHONUNBUFFERED=1 python scripts/prepare_dataset.py \
  --subsets aln_Latn als_Latn \
  --output-dir data/alb_en \
  --data-backend rows-api \
  --max-samples-per-subset 50000 \
  --rows-api-page-size 100 \
  --rows-api-retries 8 \
  --rows-api-retry-wait-seconds 2.0 \
  --rows-api-request-interval-seconds 0.15 \
  --drop-early-stop
```

## 5) Start Training (Cool + Stable Defaults)

```bash
%cd /content/drive/MyDrive/Translator
!PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python scripts/train_albanian_to_english.py \
  --data-dir data/alb_en \
  --model-name Helsinki-NLP/opus-mt-sq-en \
  --output-dir outputs/opusmt-alb-en-colab \
  --num-train-epochs 1.0 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --eval-steps 1500 \
  --save-steps 1500 \
  --logging-steps 50 \
  --fp16 \
  --generation-max-length 192 \
  --generation-num-beams 1 \
  --dataloader-num-workers 0
```

## 6) Resume After Colab Disconnect

Use the new resume flag:

```bash
%cd /content/drive/MyDrive/Translator
!PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python scripts/train_albanian_to_english.py \
  --data-dir data/alb_en \
  --model-name Helsinki-NLP/opus-mt-sq-en \
  --output-dir outputs/opusmt-alb-en-colab \
  --num-train-epochs 1.0 \
  --per-device-train-batch-size 4 \
  --per-device-eval-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --eval-steps 1500 \
  --save-steps 1500 \
  --logging-steps 50 \
  --fp16 \
  --generation-max-length 192 \
  --generation-num-beams 1 \
  --dataloader-num-workers 0 \
  --resume-from-checkpoint outputs/opusmt-alb-en-colab/checkpoint-1500
```

Adjust the checkpoint path to your latest saved checkpoint.

## 7) Use Best Checkpoint for API/UI

After training, set model path to best/final checkpoint and run server:

```bash
%cd /content/drive/MyDrive/Translator
!export TRANSLATOR_MODEL_DIR=outputs/opusmt-alb-en-colab/final && uvicorn app.server:app --host 0.0.0.0 --port 8000
```

In Colab, use the generated public URL for testing.
