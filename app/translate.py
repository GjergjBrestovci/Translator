#!/usr/bin/env python3
"""CLI tool for Albanian ↔ English translation using a fine-tuned model."""

import argparse
import sys
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def translate(text: str, tokenizer, model, max_length: int = 512) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    outputs = model.generate(**inputs, max_new_tokens=max_length, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate Albanian → English")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/opusmt-alb-en-ft-3ep-full/final",
        help="Path to fine-tuned model directory",
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
        print(f"Model not found at {model_dir}", file=sys.stderr)
        print("Train a model first or pass --model-dir to a valid checkpoint.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {model_dir}…", file=sys.stderr)
    tokenizer, model = load_model(model_dir)
    print("Ready.\n", file=sys.stderr)

    if args.text:
        source = " ".join(args.text)
        result = translate(source, tokenizer, model, args.max_length)
        print(result)
    else:
        print("Interactive mode — type Albanian text, get English back. Ctrl+C to quit.\n", file=sys.stderr)
        try:
            while True:
                source = input("sq> ").strip()
                if not source:
                    continue
                result = translate(source, tokenizer, model, args.max_length)
                print(f"en> {result}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!", file=sys.stderr)


if __name__ == "__main__":
    main()
