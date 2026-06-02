"""
Forced-choice long-context passkey evaluation for NSA checkpoints.

This script is intentionally lightweight: it does not test open-ended
generation.  Instead, it inserts a single-token keyword into a long synthetic
context and checks whether the model assigns the highest next-token score to
the correct keyword at the final query position.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from native_sparse_attention import NSAConfig, NSAForCausalLM  # noqa: E402


WORD_POOL = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "silver",
    "golden",
    "copper",
    "forest",
    "river",
    "mountain",
    "ocean",
    "desert",
    "garden",
    "castle",
    "planet",
    "winter",
    "summer",
    "thunder",
    "shadow",
    "crystal",
    "button",
    "window",
    "signal",
    "anchor",
    "bridge",
    "harbor",
    "lantern",
    "rocket",
]


FILLER_TEXT = (
    " The archive contains ordinary background text about history, science, "
    "music, weather, cities, books, rivers, gardens, and simple daily events."
)

QUERY_TEXT = " Question: What is the secret keyword? Answer: The secret keyword is"


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def single_token_candidates(tokenizer, count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for word in WORD_POOL:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) != 1:
            continue
        token_id = int(ids[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        candidates.append({"word": word, "token_id": token_id})
        if len(candidates) >= count:
            break
    if len(candidates) < count:
        raise RuntimeError(f"Only found {len(candidates)} single-token candidates; need {count}.")
    return candidates


def repeat_to_length(tokens: list[int], length: int) -> list[int]:
    if length <= 0:
        return []
    repeats = (length + len(tokens) - 1) // len(tokens)
    return (tokens * repeats)[:length]


def make_prompt(
    tokenizer,
    context_len: int,
    depth: float,
    keyword: str,
) -> tuple[list[int], dict[str, Any]]:
    needle = tokenizer.encode(f" The secret keyword is {keyword}.", add_special_tokens=False)
    query = tokenizer.encode(QUERY_TEXT, add_special_tokens=False)
    filler = tokenizer.encode(FILLER_TEXT, add_special_tokens=False)
    available = context_len - len(needle) - len(query)
    if available < 16:
        raise ValueError(
            f"context_len={context_len} is too short for needle/query "
            f"(needle={len(needle)}, query={len(query)})."
        )
    prefix_len = int(round(available * depth))
    prefix_len = max(0, min(prefix_len, available))
    suffix_len = available - prefix_len
    input_ids = repeat_to_length(filler, prefix_len) + needle + repeat_to_length(filler, suffix_len) + query
    assert len(input_ids) == context_len
    meta = {
        "needle_token_start": prefix_len,
        "needle_token_end": prefix_len + len(needle),
        "query_tokens": len(query),
        "needle_tokens": len(needle),
        "actual_depth": round(prefix_len / max(available, 1), 6),
    }
    return input_ids, meta


def load_model(args, device: torch.device, tokenizer):
    with open(args.config, encoding="utf-8") as handle:
        cfg_dict = json.load(handle)
    tok_vocab = getattr(tokenizer, "vocab_size", None)
    if tok_vocab and tok_vocab != cfg_dict.get("vocab_size"):
        cfg_dict["vocab_size"] = ((int(tok_vocab) + 63) // 64) * 64
    if args.max_position_embeddings > 0:
        cfg_dict["max_position_embeddings"] = max(
            int(cfg_dict.get("max_position_embeddings", 0) or 0),
            int(args.max_position_embeddings),
        )
    try:
        from flash_attn import flash_attn_func  # noqa: F401
    except ImportError:
        if cfg_dict.get("window_size", 0) > 0:
            print(f"[warn] flash_attn not found -> overriding window_size {cfg_dict['window_size']} -> 0")
            cfg_dict["window_size"] = 0
    cfg_clean = {key: value for key, value in cfg_dict.items() if not key.startswith("_")}
    model = NSAForCausalLM(NSAConfig(**cfg_clean)).to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg_dict


@torch.inference_mode()
def score_prompt(model, input_ids: list[int], candidate_ids: list[int], device: torch.device) -> tuple[int, list[float]]:
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        out = model(input_ids=ids, use_cache=False, logits_to_keep=1)
    logits = out.logits[0, -1, candidate_ids].float()
    pred_idx = int(torch.argmax(logits).item())
    return pred_idx, [float(value) for value in logits.detach().cpu().tolist()]


def summarize(rows: list[dict[str, Any]], lengths: list[int], depths: list[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_samples": len(rows),
        "overall_accuracy": sum(row["correct"] for row in rows) / max(len(rows), 1),
        "by_length": {},
        "by_depth": {},
        "by_length_depth": {},
    }
    for length in lengths:
        subset = [row for row in rows if row["context_len"] == length and "error" not in row]
        summary["by_length"][str(length)] = {
            "samples": len(subset),
            "accuracy": sum(row["correct"] for row in subset) / max(len(subset), 1),
            "mean_true_margin": sum(row["true_margin"] for row in subset) / max(len(subset), 1),
        }
    for depth in depths:
        subset = [row for row in rows if abs(row["depth"] - depth) < 1e-9 and "error" not in row]
        summary["by_depth"][str(depth)] = {
            "samples": len(subset),
            "accuracy": sum(row["correct"] for row in subset) / max(len(subset), 1),
            "mean_true_margin": sum(row["true_margin"] for row in subset) / max(len(subset), 1),
        }
    for length in lengths:
        for depth in depths:
            subset = [
                row
                for row in rows
                if row["context_len"] == length and abs(row["depth"] - depth) < 1e-9 and "error" not in row
            ]
            summary["by_length_depth"][f"{length}@{depth}"] = {
                "samples": len(subset),
                "accuracy": sum(row["correct"] for row in subset) / max(len(subset), 1),
                "mean_true_margin": sum(row["true_margin"] for row in subset) / max(len(subset), 1),
            }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Forced-choice passkey retrieval for NSA checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_path", default="gpt2")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--lengths", default="4096,8192,16384")
    parser.add_argument("--depths", default="0.1,0.5,0.9")
    parser.add_argument("--samples_per_cell", type=int, default=20)
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_position_embeddings", type=int, default=16384)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lengths = parse_csv_ints(args.lengths)
    depths = parse_csv_floats(args.depths)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "passkey_samples.jsonl"
    summary_path = out_dir / "passkey_summary.json"
    samples_path.write_text("", encoding="utf-8")

    tokenizer = build_tokenizer(args.tokenizer_path)
    candidates = single_token_candidates(tokenizer, args.candidate_count)
    candidate_ids = [item["token_id"] for item in candidates]
    model, cfg_dict = load_model(args, device, tokenizer)

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    for context_len in lengths:
        for depth in depths:
            for sample_idx in range(args.samples_per_cell):
                true_idx = random.randrange(len(candidates))
                true_word = candidates[true_idx]["word"]
                row: dict[str, Any] = {
                    "run_name": args.run_name,
                    "context_len": context_len,
                    "depth": depth,
                    "sample_idx": sample_idx,
                    "true_word": true_word,
                    "true_token_id": candidates[true_idx]["token_id"],
                }
                try:
                    prompt_ids, meta = make_prompt(tokenizer, context_len, depth, true_word)
                    pred_idx, scores = score_prompt(model, prompt_ids, candidate_ids, device)
                    sorted_scores = sorted(scores, reverse=True)
                    true_score = scores[true_idx]
                    max_other = max(score for idx, score in enumerate(scores) if idx != true_idx)
                    row.update(meta)
                    row.update(
                        {
                            "pred_word": candidates[pred_idx]["word"],
                            "pred_token_id": candidates[pred_idx]["token_id"],
                            "correct": int(pred_idx == true_idx),
                            "true_score": true_score,
                            "true_margin": true_score - max_other,
                            "true_rank": 1 + sum(score > true_score for score in scores),
                            "candidate_scores": {
                                candidates[idx]["word"]: round(float(score), 6) for idx, score in enumerate(scores)
                            },
                        }
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    row.update({"correct": 0, "error": repr(exc)})
                append_jsonl(samples_path, row)
                rows.append(row)
                done = len(rows)
                total = len(lengths) * len(depths) * args.samples_per_cell
                if done % max(1, args.samples_per_cell) == 0:
                    print(f"[passkey] {done}/{total} context={context_len} depth={depth} elapsed={time.time()-t0:.1f}s")

    result = {
        "run_name": args.run_name,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "tokenizer_path": args.tokenizer_path,
        "lengths": lengths,
        "depths": depths,
        "samples_per_cell": args.samples_per_cell,
        "candidate_count": args.candidate_count,
        "candidate_words": [item["word"] for item in candidates],
        "config_summary": {
            "block_counts": cfg_dict.get("block_counts"),
            "budget_mode": cfg_dict.get("budget_mode"),
            "max_position_embeddings": cfg_dict.get("max_position_embeddings"),
            "block_size": cfg_dict.get("block_size"),
        },
        "elapsed": round(time.time() - t0, 3),
        "summary": summarize(rows, lengths, depths),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
