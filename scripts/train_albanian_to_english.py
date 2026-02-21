import argparse
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/alb_en"))
    parser.add_argument("--model-name", type=str, default="Helsinki-NLP/opus-mt-sq-en")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/opusmt-alb-en-ft"))
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--save-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--generation-max-length", type=int, default=192)
    parser.add_argument("--generation-num-beams", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--eval-accumulation-steps", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--filter-noisy-pairs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-source-chars", type=int, default=20)
    parser.add_argument("--min-target-chars", type=int, default=10)
    parser.add_argument("--max-source-chars", type=int, default=1200)
    parser.add_argument("--max-target-chars", type=int, default=1200)
    parser.add_argument("--min-length-ratio", type=float, default=0.4)
    parser.add_argument("--max-length-ratio", type=float, default=2.5)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    data_files = {
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
        "test": str(args.data_dir / "test.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)

    if args.filter_noisy_pairs:
        before_sizes = {split: len(dataset[split]) for split in dataset.keys()}

        def keep_pair(example: dict) -> bool:
            source = (example.get("source") or "").strip()
            target = (example.get("target") or "").strip()
            source_len = len(source)
            target_len = len(target)
            if source_len < args.min_source_chars or target_len < args.min_target_chars:
                return False
            if source_len > args.max_source_chars or target_len > args.max_target_chars:
                return False
            ratio = target_len / max(source_len, 1)
            if ratio < args.min_length_ratio or ratio > args.max_length_ratio:
                return False
            return True

        dataset = dataset.filter(keep_pair, desc="Filtering noisy pairs")
        after_sizes = {split: len(dataset[split]) for split in dataset.keys()}
        print("Dataset filtering:")
        print({"before": before_sizes, "after": after_sizes})
    else:
        print("Dataset filtering disabled. Using full splits:")
        print({split: len(dataset[split]) for split in dataset.keys()})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    def preprocess(examples: dict) -> dict:
        model_inputs = tokenizer(
            examples["source"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["target"],
            max_length=args.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [prediction.strip() for prediction in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]

        result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        decoded_labels_flat = [item[0] for item in decoded_labels]
        chrf = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels_flat)
        result["chrf"] = chrf.get("score", 0.0)
        prediction_lens = [np.count_nonzero(prediction != tokenizer.pad_token_id) for prediction in predictions]
        result["gen_len"] = float(np.mean(prediction_lens))
        result = {key: round(value, 4) if isinstance(value, (int, float)) else value for key, value in result.items()}
        return result

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_accumulation_steps=args.eval_accumulation_steps,
        load_best_model_at_end=True,
        metric_for_best_model="score",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        fp16=args.fp16,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")

    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))

    print("Test metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()
