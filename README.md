# Qwen3.8-27B benchmarks — RTX PRO 6000 96 GB

Local benchmarks run on 2026-08-25 with one NVIDIA RTX PRO 6000 Blackwell Workstation GPU (96 GB), Windows + WSL Ubuntu-E. Model files were stored on the Ubuntu ext4 volume at `/root/llm-cache/qwen3.8-27b`.

> The figures below are single-user measurements, not a multi-user serving benchmark. Text E2E means the full non-streaming API request. Decode is generation-only server timing. The original PDF/images are deliberately not included in this repository.

## Summary: text generation

The probe requests 110 completion tokens from a 211-token prompt at `temperature=0`, after warm-up. Historical vLLM results are retained for comparison but not every older configuration was re-run in this pass.

| Runtime / configuration | Model format | Context | Text E2E tok/s | Decode tok/s | VRAM after load | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vLLM, regular graph mode | BF16 | 32K | ~27–28 | — | — | Earlier baseline, before MTP |
| vLLM, `--enforce-eager` | BF16 | 32K | 15.4 | — | — | Earlier baseline; eager is materially slower |
| **vLLM + MTP** | BF16 | 32K | **~54–57** | — | — | MTP speculative decoding, 3 draft tokens; best response speed measured |
| llama-server, UD-Q6_K_XL | mixed GGUF | 32K | **45.66** | 49.08 | 31.5 GB | Flash Attention on; full GPU offload |
| llama-server, UD-Q8_K_XL | mixed GGUF | 64K | 39.48 | 42.43 | 38.4 GB | Flash Attention on; full GPU offload |

## Summary: multimodal OCR / document understanding

Source workload: a 16-page Thai regulatory PDF rendered into page images. The task asks the model to identify the announcement, headings, important effective dates, and final-page notes. `OCR` here means visual-language extraction and understanding, not a character-OCR engine.

| Runtime / configuration | Image pages | Prompt tokens | Output tokens | E2E time | E2E tok/s | Prompt processing | Decode tok/s | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM + MTP, BF16 | 8 | 8,127 | 677 | 14.15 s | 47.85 | — | — | Earlier run; warm image cache |
| vLLM + MTP, BF16 | 16 | 16,110 | 675 | 18.50 s | 36.48 | 1,610.8 tok/s | — | Earlier run; 50% image-cache reuse |
| llama-server UD-Q6_K_XL | 1 | 2,524 | 257 | 13.66 s | 18.82 | — | — | Summary produced but dates were confused |
| llama-server UD-Q6_K_XL | 8 | 29,079 | 309 | 36.23 s | 8.53 | 1,501.06 tok/s | 42.16 | Correct document structure and main effective date |
| llama-server UD-Q8_K_XL | 1 | 2,524 | 400 | 14.76 s | 27.11 | — | — | Hit 400-token limit |
| llama-server UD-Q8_K_XL | 8 | 29,079 | 331 | 30.23 s | 10.95 | 1,642.47 tok/s | 38.11 | Correct document number/title and main dates |
| llama-server UD-Q8_K_XL | **16** | **59,445** | **388** | **59.08 s** | **6.57** | **1,559.48 tok/s** | **36.91** | Completed full 16-page structured extraction at 64K |

The vLLM vision values are not strict apples-to-apples comparisons: vLLM was capped at 1 MP/page while llama.cpp received the original full-size images, so llama.cpp used substantially more image/prompt tokens.

## Additional OCR profiles (llama-server UD-Q8_K_XL, 64K)

| OCR profile | Page | Prompt / output tokens | E2E | Decode | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Cover-letter field extraction | 1 | 2,507 / 167 | 48.43 s (3.45 tok/s) | 40.55 tok/s | Correctly separated cover-letter reference and announcement number, but swapped the gazette-publication and effective dates. **Fail for exact legal dates without verification.** |
| Complex table + legend extraction | 16 | 3,882 / 206 | 12.10 s (17.02 tok/s) | 42.61 tok/s | Correctly extracted section 5.3.6 and the table rows. Legends were nearly correct but dropped one legally meaningful word (`ถือ`). **Needs image-based verification.** |

## How llama.cpp was run

```bash
llama-server \
  -m /root/llm-cache/qwen3.8-27b/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  -mm /root/llm-cache/qwen3.8-27b/mmproj-BF16.gguf \
  -ngl 99 -fa on -c 65536 -b 2048 -ub 512 \
  --jinja --mmproj-offload --host 127.0.0.1 --port 8082
```

- `-ngl 99` and `--mmproj-offload` put all model and vision-projector layers on the GPU.
- `-fa on` explicitly enables Flash Attention.
- llama.cpp ignored the model's final MTP-only tensor block, so this GGUF path did **not** use Qwen MTP speculative decoding. This is the main reason vLLM + MTP remained faster.
- 8 full-size pages used about 29K prompt tokens; a 16-page request therefore needs 64K context.

## Important GGUF detail

The `UD` files are Unsloth Dynamic mixed-quant GGUFs, not simple uniform Q6/Q8 formats. The filenames are retained below, but llama.cpp reports these metadata values:

| File | File size | llama.cpp-reported ftype |
| --- | ---: | --- |
| `Qwen3.8-27B-UD-Q6_K_XL.gguf` | 25.91 GB | `Q4_K - Small` |
| `Qwen3.8-27B-UD-Q8_K_XL.gguf` | 31.45 GB | `Q4_K - Medium` |

## Practical recommendation

1. Use **vLLM BF16 + MTP** as the main service when interactive speed and multi-user throughput matter.
2. Use **llama-server UD-Q6_K_XL** for a smaller GGUF deployment: it used ~7 GB less VRAM and decoded about 16% faster than UD-Q8 in this test.
3. Use **llama-server UD-Q8_K_XL at 64K** when a single request must include 16 full-size pages. It completed the 59K-token request, but no clear OCR-quality advantage over UD-Q6 was established here.
4. For legal/regulatory workflows, use page-by-page extraction, page citations, and a second verification pass. A VLM summary is not certified character-perfect OCR.

## Cold-load time observed

| Model | Time until `llama-server` reported `model loaded` |
| --- | ---: |
| UD-Q6_K_XL | 17 min 15 s |
| UD-Q8_K_XL | 6 min 35 s |

This variance was storage-cache/I/O dependent and is separate from inference speed. Keeping files on Ubuntu-E ext4 avoids the Windows-mounted path, but the first GGUF read still takes time.
