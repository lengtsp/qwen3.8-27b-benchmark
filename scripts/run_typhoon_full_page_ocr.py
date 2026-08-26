#!/usr/bin/env python3
"""Run prompt-specific Typhoon OCR directly on complete document pages.

No Qwen request, layout detector, fixed crop, or PDF text layer is used.  Each
``page-*.png`` is independently resized to Typhoon's documented 1,800-pixel
long-side policy and sent to an OpenAI-compatible vLLM endpoint.  The JSON
record keeps source and sent image dimensions, timing, token usage and output
so a page-by-page quality review remains possible.
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

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_from_typhoon_readme(path: Path) -> str:
    """Extract the required upstream OCR prompt without duplicating it here."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r'prompt\s*=\s*"""(.*?)"""', text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find Typhoon's triple-quoted prompt in {path}")
    return match.group(1).strip()


def resize_for_typhoon(image: Image.Image, target_long_side: int) -> Image.Image:
    """Mirror Typhoon OCR 1.5's supplied image-sizing policy."""
    image = image.convert("RGB")
    long_side = max(image.size)
    if long_side <= 300:
        return image
    scale = target_long_side / long_side
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def image_data_url(image: Image.Image) -> str:
    from io import BytesIO

    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--typhoon-model-readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8091/v1/chat/completions")
    parser.add_argument("--model", default="typhoon-ocr-1-5")
    parser.add_argument("--target-long-side", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    if args.target_long_side < 1 or args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError("target image size, max tokens and timeout must be positive")
    pages = sorted(args.input_dir.glob("page-*.png"))
    if not pages:
        raise FileNotFoundError(f"No page-*.png images under {args.input_dir}")
    prompt = prompt_from_typhoon_readme(args.typhoon_model_readme)
    results: list[dict[str, Any]] = []

    for page_number, page_path in enumerate(pages, start=1):
        with Image.open(page_path) as source_file:
            source = source_file.convert("RGB")
        prepared = resize_for_typhoon(source, args.target_long_side)
        payload = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(prepared)}},
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
            raise RuntimeError(f"No completion choices returned for {page_path}")
        results.append(
            {
                "page": page_number,
                "source": str(page_path),
                "source_sha256": sha256_file(page_path),
                "source_dimensions": list(source.size),
                "sent_dimensions": list(prepared.size),
                "elapsed_seconds": round(elapsed, 6),
                "usage": response.get("usage"),
                "finish_reason": choices[0].get("finish_reason"),
                "text": choices[0]["message"]["content"],
            }
        )
        print(
            json.dumps(
                {
                    "page": page_number,
                    "elapsed_seconds": round(elapsed, 3),
                    "finish_reason": choices[0].get("finish_reason"),
                    "usage": response.get("usage"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    output = {
        "mode": "typhoon_pure_full_page",
        "source": {
            "input_dir": str(args.input_dir),
            "page_count": len(pages),
            "used_qwen": False,
            "used_layout_detection": False,
            "used_crop": False,
            "used_pdf_text_layer": False,
        },
        "prompt": {
            "source": str(args.typhoon_model_readme),
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        },
        "request": {
            "endpoint": args.endpoint,
            "model": args.model,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
            "target_long_side": args.target_long_side,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    total_time = sum(item["elapsed_seconds"] for item in results)
    total_output = sum((item.get("usage") or {}).get("completion_tokens", 0) for item in results)
    print(
        json.dumps(
            {
                "pages": len(results),
                "total_elapsed_seconds": round(total_time, 3),
                "total_completion_tokens": total_output,
                "aggregate_output_tokens_per_second": round(total_output / total_time, 3),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
