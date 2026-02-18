import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm


def iter_subset(subset: str, limit: int | None) -> Iterable[dict]:
    ds = load_dataset(
        "HuggingFaceFW/finetranslations",
        name=subset,
        split="train",
        streaming=True,
    )

    for index, row in enumerate(ds):
        if limit is not None and index >= limit:
            break
        yield row


def iter_subset_rows_api(
    subset: str,
    limit: int | None,
    page_size: int,
    retries: int,
    retry_wait_seconds: float,
    request_interval_seconds: float,
) -> Iterable[dict]:
    base_url = "https://datasets-server.huggingface.co/rows"
    dataset_name = "HuggingFaceFW/finetranslations"

    produced = 0
    offset = 0

    while True:
        if limit is not None and produced >= limit:
            break

        current_length = page_size
        if limit is not None:
            current_length = min(current_length, limit - produced)
            if current_length <= 0:
                break

        query = urllib.parse.urlencode(
            {
                "dataset": dataset_name,
                "config": subset,
                "split": "train",
                "offset": offset,
                "length": current_length,
            }
        )
        url = f"{base_url}?{query}"

        payload = None
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except Exception:
                if attempt >= retries:
                    raise
                time.sleep(retry_wait_seconds * (attempt + 1))

        rows = payload.get("rows", []) if payload else []
        if not rows:
            break

        for wrapped_row in rows:
            row = wrapped_row.get("row", wrapped_row)
            yield row
            produced += 1
            if limit is not None and produced >= limit:
                break

        offset += len(rows)
        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["aln_Latn", "als_Latn"],
        help="FineTranslations subsets to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/alb_en"),
        help="Directory to write train/validation/test JSONL files.",
    )
    parser.add_argument(
        "--max-samples-per-subset",
        type=int,
        default=300_000,
        help="Max records to stream per subset. Use -1 for all.",
    )
    parser.add_argument(
        "--min-source-chars",
        type=int,
        default=20,
        help="Minimum source characters to keep.",
    )
    parser.add_argument(
        "--drop-early-stop",
        action="store_true",
        help="Drop rows where early_stop is true.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.98)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument(
        "--data-backend",
        choices=["streaming", "rows-api"],
        default="streaming",
        help="Source backend: HF datasets streaming or datasets-server rows API.",
    )
    parser.add_argument(
        "--rows-api-page-size",
        type=int,
        default=100,
        help="Rows API page size when --data-backend rows-api is used.",
    )
    parser.add_argument(
        "--rows-api-retries",
        type=int,
        default=5,
        help="Rows API retries per page.",
    )
    parser.add_argument(
        "--rows-api-retry-wait-seconds",
        type=float,
        default=1.0,
        help="Base wait between rows API retries.",
    )
    parser.add_argument(
        "--rows-api-request-interval-seconds",
        type=float,
        default=0.0,
        help="Wait time between successful rows API page requests.",
    )
    args = parser.parse_args()

    max_per_subset = None if args.max_samples_per_subset == -1 else args.max_samples_per_subset

    rows: list[dict] = []
    for subset in args.subsets:
        print(f"Streaming subset: {subset}")
        if args.data_backend == "rows-api":
            iterator = iter_subset_rows_api(
                subset=subset,
                limit=max_per_subset,
                page_size=args.rows_api_page_size,
                retries=args.rows_api_retries,
                retry_wait_seconds=args.rows_api_retry_wait_seconds,
                request_interval_seconds=args.rows_api_request_interval_seconds,
            )
        else:
            iterator = iter_subset(subset, max_per_subset)

        for row in tqdm(iterator, desc=subset):
            source = (row.get("og_full_text") or "").strip()
            target = (row.get("translated_text") or "").strip()
            if not source or not target:
                continue
            if len(source) < args.min_source_chars:
                continue
            if args.drop_early_stop and bool(row.get("early_stop", False)):
                continue

            rows.append(
                {
                    "source": source,
                    "target": target,
                    "subset": subset,
                    "id": row.get("id", ""),
                }
            )

    if not rows:
        raise RuntimeError("No rows collected. Relax filters or increase sample limit.")

    random.seed(args.seed)
    random.shuffle(rows)

    train_ratio = args.train_ratio
    valid_ratio = args.validation_ratio
    test_ratio = 1.0 - train_ratio - valid_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + validation_ratio must be < 1.0")

    n = len(rows)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_rows = rows[:n_train]
    valid_rows = rows[n_train : n_train + n_valid]
    test_rows = rows[n_train + n_valid :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "validation.jsonl", valid_rows)
    write_jsonl(args.output_dir / "test.jsonl", test_rows)

    metadata = {
        "subsets": args.subsets,
        "num_total": n,
        "num_train": len(train_rows),
        "num_validation": len(valid_rows),
        "num_test": len(test_rows),
        "data_backend": args.data_backend,
        "max_samples_per_subset": args.max_samples_per_subset,
        "min_source_chars": args.min_source_chars,
        "drop_early_stop": args.drop_early_stop,
        "seed": args.seed,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("Dataset prepared:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
