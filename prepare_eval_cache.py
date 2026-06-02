"""Prepare reusable validation/test token caches for large-data experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from simple_train import build_eval_cache, build_tokenizer, optional_str


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--tokenizer_path", default="gpt2")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--text_column", default="text")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--cache_seed", type=int, default=0)
    parser.add_argument("--train_split", default="")
    parser.add_argument("--train_data_files", default="")
    parser.add_argument("--train_tokens", type=int, default=0)
    parser.add_argument("--train_skip_tokens", type=int, default=0)
    parser.add_argument("--val_split", required=True)
    parser.add_argument("--val_data_files", default="")
    parser.add_argument("--val_tokens", type=int, required=True)
    parser.add_argument("--val_skip_tokens", type=int, default=0)
    parser.add_argument("--test_split", required=True)
    parser.add_argument("--test_data_files", default="")
    parser.add_argument("--test_tokens", type=int, required=True)
    parser.add_argument("--test_skip_tokens", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = build_tokenizer(args.tokenizer_path)
    dataset_name = optional_str(args.dataset_name)
    cache_dir = Path(args.cache_dir)

    train_meta = None
    if args.train_split and args.train_tokens > 0:
        train_meta = build_eval_cache(
            dataset=args.dataset,
            dataset_name=dataset_name,
            split=args.train_split,
            data_files=optional_str(args.train_data_files),
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer_path,
            seq_len=args.seq_len,
            target_tokens=args.train_tokens,
            skip_tokens=args.train_skip_tokens,
            text_column=args.text_column,
            cache_dir=cache_dir,
            cache_seed=args.cache_seed,
        )

    val_meta = build_eval_cache(
        dataset=args.dataset,
        dataset_name=dataset_name,
        split=args.val_split,
        data_files=optional_str(args.val_data_files),
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        target_tokens=args.val_tokens,
        skip_tokens=args.val_skip_tokens,
        text_column=args.text_column,
        cache_dir=cache_dir,
        cache_seed=args.cache_seed,
    )
    test_meta = build_eval_cache(
        dataset=args.dataset,
        dataset_name=dataset_name,
        split=args.test_split,
        data_files=optional_str(args.test_data_files),
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer_path,
        seq_len=args.seq_len,
        target_tokens=args.test_tokens,
        skip_tokens=args.test_skip_tokens,
        text_column=args.text_column,
        cache_dir=cache_dir,
        cache_seed=args.cache_seed,
    )

    summary = {"train": train_meta, "validation": val_meta, "test": test_meta}
    summary_path = cache_dir / "cache_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    # Some streaming dataset backends leave downloader threads alive and can
    # crash during interpreter finalization after all caches are safely written.
    os._exit(0)


if __name__ == "__main__":
    main()
