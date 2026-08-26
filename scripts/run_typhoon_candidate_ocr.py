#!/usr/bin/env python3
"""Read dynamically detected layout crops with a running Typhoon OCR server.

The input is a ``candidate-manifest.json`` created by
``build_auto_layout_ocr_candidates.py``.  This client intentionally produces
an evidence record only: it does not edit Qwen's canonical full-page OCR text
or decide that a candidate is correct.  Use an independently approved Typhoon
OCR prompt file because Typhoon OCR 1.5 is prompt-specific.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        suffix
    )
    if mime_type is None:
        raise ValueError(f"Unsupported candidate image type: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def post_json(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Typhoon API returned HTTP {error.code}: {body}") from error


def prompt_from_typhoon_readme(path: Path) -> str:
    """Read the upstream, prompt-specific OCR prompt without copying it here."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'prompt\s*=\s*"""(.*?)"""', text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find Typhoon's triple-quoted prompt in {path}")
    return match.group(1).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    prompt_source = parser.add_mutually_exclusive_group(required=True)
    prompt_source.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing Typhoon OCR's documented prompt verbatim.",
    )
    prompt_source.add_argument(
        "--typhoon-model-readme",
        type=Path,
        help="Typhoon model README; extracts its documented OCR prompt without duplicating it.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8091/v1/chat/completions")
    parser.add_argument("--model", default="typhoon-ocr-1-5")
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="Run an exact candidate id; repeatable. Omit to run every candidate.",
    )
    parser.add_argument(
        "--page",
        action="append",
        type=int,
        default=[],
        help="Run every candidate from this page; repeatable. Omit to use every page.",
    )
    args = parser.parse_args()

    if args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError("--max-tokens and --timeout must be positive")
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        prompt_source_record = str(args.prompt_file)
    else:
        prompt = prompt_from_typhoon_readme(args.typhoon_model_readme)
        prompt_source_record = str(args.typhoon_model_readme)
    if not prompt:
        raise ValueError("--prompt-file is empty")
    source_manifest = json.loads(args.candidate_manifest.read_text(encoding="utf-8"))
    allowed_ids = set(args.candidate_id)
    allowed_pages = set(args.page)
    manifest_root = args.candidate_manifest.parent
    results: list[dict[str, Any]] = []

    for page in source_manifest["pages"]:
        page_number = int(page["page"])
        if allowed_pages and page_number not in allowed_pages:
            continue
        for candidate in page["candidates"]:
            candidate_id = str(candidate["id"])
            if allowed_ids and candidate_id not in allowed_ids:
                continue
            image_path = manifest_root / candidate["image"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url(image_path)}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": args.max_tokens,
            }
            started = time.perf_counter()
            response = post_json(args.endpoint, payload, args.timeout)
            elapsed = time.perf_counter() - started
            choices = response.get("choices", [])
            if not choices:
                raise RuntimeError(f"No completion choices returned for page {page_number}, {candidate_id}")
            results.append(
                {
                    "page": page_number,
                    "candidate": candidate,
                    "elapsed_seconds": round(elapsed, 6),
                    "usage": response.get("usage"),
                    "finish_reason": choices[0].get("finish_reason"),
                    "text": choices[0]["message"]["content"],
                }
            )

    if not results:
        raise RuntimeError("No candidates matched --page/--candidate-id filters")
    output = {
        "source_candidate_manifest": str(args.candidate_manifest),
        "prompt_source": prompt_source_record,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "request": {
            "endpoint": args.endpoint,
            "model": args.model,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
        },
        "warning": "Evidence only: do not automatically replace canonical full-page OCR with this output.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"requests": len(results), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
