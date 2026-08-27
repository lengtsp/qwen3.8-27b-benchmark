#!/usr/bin/env python3
"""Persist the Paddle document models used by the hybrid OCR workflow."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# ``huggingface_hub.constants`` reads these at import time.
# Xet's default adaptive setting can fall back to one slow range request on
# WSL.  Keep downloads resumable while still allowing an explicit caller
# override for a deliberately rate-limited network.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "16")

from huggingface_hub import snapshot_download


MODELS = {
    "paddleocr-vl-1.6": (
        "PaddlePaddle/PaddleOCR-VL-1.6",
        "PaddleOCR-VL-1.6",
    ),
    "pp-doclayoutv3": (
        "PaddlePaddle/PP-DocLayoutV3",
        "PP-DocLayoutV3",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("/root/llm-cache"))
    parser.add_argument(
        "--attempts",
        type=int,
        default=5,
        help="Retry a resumable snapshot when the Xet CDN ends a range transfer.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        action="append",
        default=[],
        help="Repeat to limit downloads; omit to download both models.",
    )
    args = parser.parse_args()
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1")
    requested = args.model or list(MODELS)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    for name in requested:
        repo_id, directory_name = MODELS[name]
        destination = args.cache_root / directory_name
        print(f"START {repo_id} -> {destination}", flush=True)
        for attempt in range(1, args.attempts + 1):
            try:
                path = snapshot_download(repo_id=repo_id, local_dir=destination)
                break
            except Exception as error:
                if attempt == args.attempts:
                    raise
                delay_seconds = min(60, 3 * 2 ** (attempt - 1))
                print(
                    f"RETRY {attempt}/{args.attempts} after {type(error).__name__}: "
                    f"{error}; waiting {delay_seconds}s",
                    flush=True,
                )
                time.sleep(delay_seconds)
        print(f"COMPLETE {repo_id} -> {path}", flush=True)


if __name__ == "__main__":
    main()
