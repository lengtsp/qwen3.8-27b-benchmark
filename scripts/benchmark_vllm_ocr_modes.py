#!/usr/bin/env python3
"""Benchmark Qwen VLM OCR one page at a time versus pages in one request."""

import argparse
import base64
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image


def image_data_url(path: Path, cache_buster_id: int) -> str:
    """Return a PNG data URL, optionally with an invisible cache-busting pixel.

    The lower-left page margin is white in the rendered test PDF.  Changing one
    pixel makes the batch request a distinct multimodal input from the
    page-by-page request while leaving the readable document content unchanged.
    """
    if cache_buster_id == 0:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    with Image.open(path) as source:
        image = source.convert("RGB")
    image.putpixel(
        (cache_buster_id % 16, image.height - 1),
        (255, 255, max(240, 255 - cache_buster_id)),
    )
    encoded_image = io.BytesIO()
    image.save(encoded_image, format="PNG")
    encoded = base64.b64encode(encoded_image.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def call_api(base_url: str, model: str, content: list[dict], max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    usage = result["usage"]
    return {
        "elapsed_seconds": elapsed,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "finish_reason": result["choices"][0]["finish_reason"],
        "response_text": result["choices"][0]["message"]["content"],
    }


def single_content(
    page_number: int, image_url: str, instruction: str | None = None
) -> list[dict]:
    return [
        {"type": "text", "text": f"[เอกสารหน้า {page_number}]"},
        {"type": "image_url", "image_url": {"url": image_url}},
        {
            "type": "text",
            "text": instruction
            or (
                "ทำ OCR จากภาพนี้เท่านั้น ถอดข้อความภาษาไทยตามลำดับที่อ่านได้ "
                "โดยคงหัวข้อ เลขข้อ วันที่ และตัวเลขสำคัญไว้ให้มากที่สุด. "
                "หากอ่านไม่ชัดให้เขียน [อ่านไม่ชัด] และห้ามเติมจากความรู้ภายนอกภาพ."
            ),
        },
    ]


def batch_content(
    page_numbers: list[int], image_urls: list[str], instruction: str | None = None
) -> list[dict]:
    content: list[dict] = []
    for page_number, image_url in zip(page_numbers, image_urls, strict=True):
        content.append({"type": "text", "text": f"[เอกสารหน้า {page_number}]"})
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    content.append(
        {
            "type": "text",
            "text": instruction
            or (
                "ทำ OCR จากภาพทุกหน้าที่ให้มา ตอบแยกหัวข้อ [หน้า n] และถอดข้อความไทย "
                "ตามลำดับที่อ่านได้ของแต่ละหน้า โดยคงหัวข้อ เลขข้อ วันที่ และตัวเลขสำคัญ. "
                "หากอ่านไม่ชัดให้เขียน [อ่านไม่ชัด] และห้ามเติมจากความรู้ภายนอกภาพ."
            ),
        }
    )
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("sequential", "batch", "sharded-batch"),
        required=True,
        help="sharded-batch sends fixed-size groups while recording each request separately.",
    )
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--max-tokens-per-page", type=int, default=700)
    parser.add_argument(
        "--instruction",
        help="Optional image-only OCR instruction; use a narrow field request for verification crops.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Images per request when --mode sharded-batch is selected.",
    )
    parser.add_argument(
        "--cache-buster-id",
        type=int,
        default=0,
        help="Positive id makes a visually identical but cache-distinct image set.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in args.images:
        if not path.is_file():
            raise FileNotFoundError(path)
    page_numbers = list(range(args.first_page, args.first_page + len(args.images)))

    if args.mode == "sequential":
        results = []
        for page_number, path in zip(page_numbers, args.images, strict=True):
            results.append(
                {
                    "page": page_number,
                    "image": str(path),
                    **call_api(
                        args.base_url,
                        args.model,
                        single_content(
                            page_number,
                            image_data_url(path, args.cache_buster_id), args.instruction
                        ),
                        args.max_tokens_per_page,
                    ),
                }
            )
    elif args.mode == "batch":
        # A distinct id makes every test mode use a separate multimodal input
        # while leaving readable page content unchanged.
        results = [
            {
                "pages": page_numbers,
                "images": [str(path) for path in args.images],
                **call_api(
                    args.base_url,
                    args.model,
                    batch_content(
                        page_numbers,
                        [
                            image_data_url(path, args.cache_buster_id)
                            for path in args.images
                        ],
                        args.instruction,
                    ),
                    args.max_tokens_per_page * len(args.images),
                ),
            }
        ]
    else:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        results = []
        for offset in range(0, len(args.images), args.batch_size):
            batch_paths = args.images[offset : offset + args.batch_size]
            batch_pages = page_numbers[offset : offset + args.batch_size]
            results.append(
                {
                    "pages": batch_pages,
                    "images": [str(path) for path in batch_paths],
                    **call_api(
                        args.base_url,
                        args.model,
                        batch_content(
                            batch_pages,
                            [
                                image_data_url(
                                    path, args.cache_buster_id + page_number
                                )
                                for page_number, path in zip(
                                    batch_pages, batch_paths, strict=True
                                )
                            ],
                            args.instruction,
                        ),
                        args.max_tokens_per_page * len(batch_paths),
                    ),
                }
            )

    elapsed = sum(row["elapsed_seconds"] for row in results)
    prompt_tokens = sum(row["prompt_tokens"] for row in results)
    completion_tokens = sum(row["completion_tokens"] for row in results)
    summary = {
        "mode": args.mode,
        "page_numbers": page_numbers,
        "page_count": len(args.images),
        "max_tokens_per_page": args.max_tokens_per_page,
        "cache_buster_id": args.cache_buster_id,
        "aggregate_elapsed_seconds": elapsed,
        "aggregate_prompt_tokens": prompt_tokens,
        "aggregate_completion_tokens": completion_tokens,
        "end_to_end_output_tokens_per_second": completion_tokens / elapsed,
        "seconds_per_page": elapsed / len(args.images),
        "requests": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "requests"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
