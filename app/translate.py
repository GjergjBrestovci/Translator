#!/usr/bin/env python3
"""CLI tool for Albanian ↔ English translation using a fine-tuned model."""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_dir: str):
    logger.info("Loading model from %s onto %s …", model_dir, DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.to(DEVICE)
    model.eval()
    logger.info("Model ready.")
    return tokenizer, model


def translate(text: str, tokenizer, model, max_length: int = 512) -> str:
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Albanian → English")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get(
            "TRANSLATOR_MODEL_DIR", "outputs/opusmt-alb-en-ft-3ep-full/final"
        ),
        help="Path to fine-tuned model directory (or set TRANSLATOR_MODEL_DIR env var)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length",
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to translate (omit for interactive mode)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir
    if not Path(model_dir).exists():
        logger.error("Model not found at %s", model_dir)
        logger.error("Train a model first or pass --model-dir to a valid checkpoint.")
        sys.exit(1)

    tokenizer, model = load_model(model_dir)

    if args.text:
        source = " ".join(args.text)
        result = translate(source, tokenizer, model, args.max_length)
        print(result)
    else:
        logger.info("Interactive mode — type Albanian text, get English back. Ctrl+C to quit.\n")
        try:
            while True:
                source = input("sq> ").strip()
                if not source:
                    continue
                result = translate(source, tokenizer, model, args.max_length)
                print(f"en> {result}\n")
        except (KeyboardInterrupt, EOFError):
            logger.info("Bye!")


if __name__ == "__main__":
    main()
