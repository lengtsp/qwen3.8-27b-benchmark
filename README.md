# Qwen3.8-27B benchmarks — RTX PRO 6000 96 GB

Local benchmarks run on 2026-08-25–26 with one NVIDIA RTX PRO 6000 Blackwell Workstation GPU (96 GB), Windows + WSL Ubuntu-E. Model files were stored on the Ubuntu ext4 volume at `/root/llm-cache/qwen3.8-27b`.

> The figures below are single-user measurements, not a multi-user serving benchmark. Text E2E means the full non-streaming API request. Decode is generation-only server timing. The original PDF/images are deliberately not included in this repository.

## Quick-start: recommended vLLM parameters

The commands below are separated deliberately: **vLLM regular** has no speculative decoding; **vLLM + MTP** adds Qwen's native MTP draft head. The fastest accuracy-first default for coding and page OCR is MTP 3 at 32K context with BF16 KV (`auto`).

### vLLM regular (baseline, no MTP)

```bash
vllm serve /root/llm-cache/qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --served-model-name qwen3.8-27b \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '{"image":16}' \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.90
```

### vLLM + MTP (recommended for text, code, and OCR)

```bash
vllm serve /root/llm-cache/qwen3.8-27b \
  --host 127.0.0.1 --port 8000 \
  --served-model-name qwen3.8-27b \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --limit-mm-per-prompt '{"image":16}' \
  --mm-processor-kwargs '{"max_pixels":1048576}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.90
```

The only inference-mode difference is `--speculative-config`. Do **not** add `--enforce-eager`; the tested graph-mode service reports `enforce_eager=False`.

| Situation | `--max-model-len` | `--kv-cache-dtype` | vLLM regular | vLLM + MTP (recommended) |
| --- | ---: | --- | --- | --- |
| Text/chat | 12,288 | `auto` | use the baseline command above with `12288` | `bash scripts/run_vllm_profile.sh text` |
| Coding / agent | 32,768 | `auto` | use the baseline command above | `bash scripts/run_vllm_profile.sh code` |
| OCR page-by-page | 32,768 | `auto` | use the baseline command above | `bash scripts/run_vllm_profile.sh ocr` |
| 80K long document, accuracy first | 81,920 | `auto` | use the baseline command above with `81920` | `bash scripts/run_vllm_profile.sh long-80k-quality` |
| 80K long document, more concurrent users | 81,920 | `fp8_e4m3` | use the baseline command above with `81920` / `fp8_e4m3` | `bash scripts/run_vllm_profile.sh long-80k-capacity` |
| 120K long document, accuracy first | 122,880 | `auto` | use the baseline command above with `122880` | `bash scripts/run_vllm_profile.sh long-120k-quality` |
| 120K long document, more concurrent users | 122,880 | `fp8_e4m3` | use the baseline command above with `122880` / `fp8_e4m3` | `bash scripts/run_vllm_profile.sh long-120k-capacity` |

[`scripts/run_vllm_profile.sh`](scripts/run_vllm_profile.sh) implements the MTP profiles. FP8 expands KV capacity but requires a task-quality check before production OCR.

### Metric definitions

- **E2E (end-to-end)** = completion/output tokens divided by wall-clock time from sending the HTTP request until its complete response arrives. It includes request serialization/upload, image preprocessing and vision encoding, prompt prefill, time-to-first-token, decoding, and API overhead. It does **not** include loading the model at server startup.
- **Decode tok/s** = generation phase only, when that server-side timing is available. It is normally higher than E2E and is not a substitute for the user's observed request time.
- **Cold image** below means the first request for those page-image bytes in that freshly started server. **Warm same image** intentionally reuses the same image after the kernel and multimodal cache are populated.

## Primary comparison tables

Same-input configurations are compared within each row. All single-request figures are warm E2E output tok/s and exclude model startup; they include prefill, so they are not comparable to aggregate multi-user serving throughput.

### Text, code, and OCR

| Workload | Controlled input / output | vLLM regular BF16 | MTP 1 BF16 | MTP 2 BF16 | MTP 3 BF16 | MTP 3 FP8 KV | Best measured configuration | Recommended default |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Text generation | 211-token prompt, 110-token completion | 28.71 | 44.63 | 55.78 | **60.18** | 52.42 | MTP 3 + BF16 | MTP 3, BF16 KV |
| Code generation | Deterministic Python task, maximum 384 output tokens | — | 45.26* | 59.15* | **79.74** | — | MTP 3 + BF16 | MTP 3, BF16 KV, 32K context |
| Page OCR / document understanding | New 1 MP images, pages 2–4 after page-1 warm-up, maximum 700 output tokens | 30.34 | 41.58 | 50.57 | **54.56** | 53.08 | MTP 3 + BF16 | MTP 3, BF16 KV, 32K context |

\* MTP 1 and MTP 2 code outputs hit the 384-token cap. MTP 3 stopped naturally after a complete 322-token solution, so the code row shows a useful operational result but is not a quality-normalized output-length comparison.

### Long input + long output: fixed 4,096-token completion

This is the primary "read a long document, then write a long answer" probe. Every row sends the stated **actual API prompt tokens** and receives exactly **4,096 completion tokens** (`finish_reason=length`). The server was warm, model startup/CUDA-graph capture and one discarded warm-up request are excluded, and every configuration used `max-model-len=122880` so the input plus output fits the 120K profile.

The input is deterministic synthetic retention records and the requested output is `ACK` repeated 4,096 times. It measures real API prefill plus long decode, but it is deliberately a **throughput stress test**, not a long-document factual-accuracy, coding-quality, or OCR-quality evaluation. The repeatable output is accepted at 100% by MTP, so do not assume its absolute MTP multiplier will transfer unchanged to open-ended reasoning.

| Context profile | Actual prompt tokens | Completion tokens | vLLM regular BF16 elapsed / E2E tok/s | vLLM + MTP 3 BF16 elapsed / E2E tok/s | vLLM + MTP 3 FP8 KV elapsed / E2E tok/s | Fastest vs. regular |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 3,957 | 4,096 | 141.66 s / 28.91 | 49.16 s / 83.31 | **48.44 s / 84.57** | FP8 KV, 2.93x |
| 32K | 28,543 | 4,096 | 150.31 s / 27.25 | 66.24 s / 61.84 | **53.75 s / 76.21** | FP8 KV, 2.80x |
| 80K | 77,658 | 4,096 | 171.54 s / 23.88 | 104.36 s / 39.25 | **67.28 s / 60.88** | FP8 KV, 2.55x |
| 120K | 118,660 | 4,096 | 192.52 s / 21.28 | 141.83 s / 28.88 | **82.55 s / 49.62** | FP8 KV, 2.33x |

For this deliberately high-MTP-acceptance workload, MTP 3 is the material accelerator for long answers. FP8 KV is nearly tied with BF16 at the 8K case, but becomes progressively more valuable as the real prompt becomes longer: at 120K it reduces end-to-end time from 141.83 s (MTP 3 BF16) to 82.55 s (MTP 3 FP8 KV). Keep BF16 KV as the accuracy-first default for OCR and semantic long-document work until the specific task has passed a quality comparison; select FP8 KV for maximum 80K–120K response speed/capacity after that validation.

### Long-context text: MTP 3 with a fixed 256-token output

| Context profile | Actual prompt tokens | BF16 KV (`auto`) E2E tok/s | FP8 E4M3 KV E2E tok/s | Best for this single request | Recommended KV choice |
| --- | ---: | ---: | ---: | --- | --- |
| 8K | 7,623 | **59.55** | 54.74 | BF16 | BF16 KV |
| 12K | 11,727 | **50.05** | not repeated | BF16 | BF16 KV |
| 24K | 24,020 | **33.18** | not repeated | BF16 | BF16 KV |
| 32K | 32,209 | 26.41 | **28.43** | FP8 (small margin) | BF16 for accuracy; FP8 when capacity matters |
| 80K | 81,362 | 11.22 | **12.23** | FP8 | FP8 for concurrency; BF16 for accuracy-first work |
| 120K | 122,326 | 6.87 | **7.19** | FP8 | FP8 when more than ~3 concurrent long requests are needed |

### Context capacity: MTP 3 at `max-model-len=122880`

| KV cache | Usable KV tokens reported by vLLM | 120K context-sized requests reported by vLLM | Practical meaning |
| --- | ---: | ---: | --- |
| BF16 `auto` | 443,404 | 3.61 | Accuracy-first long-context serving |
| FP8 `fp8_e4m3` | 840,830 | 6.84 | ~1.9× the long-context capacity; validate OCR/task quality first |

## Summary: text generation

The probe requests 110 completion tokens from a 211-token prompt at `temperature=0`, after warm-up. Historical vLLM results are retained for comparison but not every older configuration was re-run in this pass.

| Runtime / configuration | Model format | Context | Text E2E tok/s | Decode tok/s | VRAM after load | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| vLLM, regular graph mode | BF16 | 32K | **28.71** | — | — | Controlled 3-run re-test on 2026-08-26; 108 completion tokens/run |
| vLLM, `--enforce-eager` | BF16 | 32K | 15.4 | — | — | Earlier baseline; eager is materially slower |
| **vLLM + MTP** | BF16 | 32K | **60.18** | — | — | Controlled 3-run re-test; MTP speculative decoding, 3 draft tokens |
| vLLM + MTP + FlashInfer FP8 KV (trial) | BF16 weights + FP8 E4M3 KV | 32K | 52.42 | — | 52.46 GB + 31.65 GB KV | Controlled 3-run re-test; increases KV capacity but is slower for this single-user short-text workload |
| llama-server, UD-Q6_K_XL | mixed GGUF | 32K | **45.66** | 49.08 | 31.5 GB | Flash Attention on; full GPU offload |
| llama-server, UD-Q8_K_XL | mixed GGUF | 64K | 39.48 | 42.43 | 38.4 GB | Flash Attention on; full GPU offload |

### vLLM MTP draft-depth sweep

All rows below use graph mode, BF16 KV, the same 211-token text prompt and a three-run aggregate after warm-up. The OCR column is the paired warm-server/cold-image protocol: discard page 1 as warm-up, then process pages 2–4 one at a time at 1 MP/page and aggregate the responses. Three draft tokens are the best setting on this GPU; setting fewer draft tokens reduces the speculative work but loses more accepted tokens than it saves.

| MTP draft tokens | Text E2E tok/s | OCR E2E tok/s | OCR aggregate (output / time) |
| ---: | ---: | ---: | --- |
| 1 | 44.63 | 41.58 | 1,005 / 24.171 s |
| 2 | 55.78 | 50.57 | 1,273 / 25.173 s |
| **3** | **60.18** | **54.56** | 1,212 / 22.213 s |

### Long-context and KV-cache sweep (MTP 3)

This warmed single-request probe holds generation at 256 output tokens (`finish_reason=length`) and sends deterministic text inputs close to the stated context size. It excludes model loading, CUDA-graph capture, and the discarded warm-up request. **E2E TPS includes prefill**, so the lower value at 80K/120K is principally the cost of reading the much longer input; it is not decode-only TPS.

| Context profile | Actual prompt tokens | BF16 KV (`auto`) elapsed / E2E TPS | FP8 E4M3 KV elapsed / E2E TPS | FP8 versus BF16 |
| --- | ---: | ---: | ---: | ---: |
| 8K | 7,623 | 4.30 s / **59.55** | 4.68 s / 54.74 | -8.1% |
| 12K | 11,727 | 5.12 s / **50.05** | not repeated | — |
| 24K | 24,020 | 7.72 s / **33.18** | not repeated | — |
| 32K | 32,209 | 9.69 s / 26.41 | 9.00 s / **28.43** | +7.7% |
| 80K | 81,362 | 22.83 s / 11.22 | 20.94 s / **12.23** | +9.0% |
| 120K | 122,326 | 37.29 s / 6.87 | 35.62 s / **7.19** | +4.7% |

At `max-model-len=122880`, vLLM reported **443,404 usable KV-cache tokens / 3.61 context-sized requests** with BF16 KV, versus **840,830 / 6.84** with `fp8_e4m3`. The major FP8 benefit is therefore capacity/concurrency, not a universal single-user speed gain. In this vLLM version, BF16 KV selected FlashAttention 2 while FP8 KV selected FlashInfer; this is the working end-to-end stack comparison.

| Workload | Recommended `max-model-len` | KV cache | MTP draft tokens | Why |
| --- | ---: | --- | ---: | --- |
| Interactive text/chat | 12,288 | `auto` (BF16) | 3 | Accuracy-first default; 50.05 E2E TPS at the 11.7K input probe. |
| Coding / agents | 32,768 | `auto` (BF16) | 3 | Enough space for source and tool output; MTP 3 produced the complete tested code response. |
| Page-at-a-time OCR | 32,768 | `auto` (BF16) | 3 | MTP 3 reached 54.56 E2E TPS in the paired 1 MP OCR test. FP8 is not the OCR default because the prior OCR trial did not improve speed and needs quality validation. |
| 80K–120K long documents | 81,920 or 122,880 | `auto` for accuracy; `fp8_e4m3` when high concurrency is needed | 3 | FP8 roughly doubles KV capacity and was modestly faster in these long-input probes. |

The runner accepts these tested values: `8192`, `12288`, `24576`, `32768`, `81920`, and `122880`. It runs with CUDA graphs (`enforce_eager=False` in the vLLM log), so no eager-mode penalty is included in the results above.

### Ready-to-run vLLM profiles

[`scripts/run_vllm_profile.sh`](scripts/run_vllm_profile.sh) encodes the recommended parameters below. It uses the local model path by default; set `MODEL=/your/model/path` to override it. Every profile uses Qwen's native MTP with **3 draft tokens**, BF16 weights, `gpu-memory-utilization=0.90`, graph mode, and a 1 MP/page vision limit. The FP8 profiles require a FlashInfer-capable vLLM install and should pass a task-quality check before production OCR use.

| Script profile | Use case | `--max-model-len` | `--kv-cache-dtype` | Command |
| --- | --- | ---: | --- | --- |
| `text` | Interactive text/chat | 12,288 | `auto` | `bash scripts/run_vllm_profile.sh text` |
| `code` | Coding and agents | 32,768 | `auto` | `bash scripts/run_vllm_profile.sh code` |
| `ocr` | Page-at-a-time OCR/document understanding | 32,768 | `auto` | `bash scripts/run_vllm_profile.sh ocr` |
| `long-80k-quality` | 80K documents, accuracy first | 81,920 | `auto` | `bash scripts/run_vllm_profile.sh long-80k-quality` |
| `long-80k-capacity` | 80K documents, higher concurrency | 81,920 | `fp8_e4m3` | `bash scripts/run_vllm_profile.sh long-80k-capacity` |
| `long-120k-quality` | 120K documents, accuracy first | 122,880 | `auto` | `bash scripts/run_vllm_profile.sh long-120k-quality` |
| `long-120k-capacity` | 120K documents, higher concurrency | 122,880 | `fp8_e4m3` | `bash scripts/run_vllm_profile.sh long-120k-capacity` |

The essential vLLM command emitted by the script is:

```bash
vllm serve "$MODEL" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --gpu-memory-utilization 0.90
```

### Concurrent serving benchmark: comparison with the supplied RTX PRO 6000 result

The supplied screenshot and the test below use a different metric from the single-user table: **output token throughput is the aggregate across concurrent requests**, not the speed one caller sees. The useful per-caller decode proxy is `1 / TPOT`. The screenshot's 41.70 ms TPOT is about 24.0 tok/s per active stream even though its aggregate output figure is 112.60 tok/s.

Both comparison workloads are random text-completion traffic: 100 requests, 1,024 input tokens/request, 256 generated tokens/request, `ignore_eos`, `request_rate=inf`, and maximum concurrency 5. This is not an OCR workload. The screenshot shows vLLM `v0.17.0`, FP8 KV cache, 32K context and 92% GPU-memory utilization, but does **not** show an MTP configuration. The local run uses vLLM `0.27.1`, BF16 KV, FlashAttention 2, and MTP with three draft tokens. The version/config difference means it is a controlled operational comparison rather than a perfectly identical binary reproduction.

| Run | Output throughput (aggregate) | Mean TTFT | P99 TTFT | Mean TPOT | Approx. per-stream decode | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Supplied RTX PRO 6000 screenshot | 112.60 tok/s | 733.18 ms | 965.00 ms | 41.70 ms | 24.0 tok/s | vLLM 0.17.0, FP8 KV; no MTP flag shown |
| Local MTP 3, first pass after service restart | 198.14 tok/s | 544.13 ms | 6,737.52 ms | 22.67 ms | 44.1 tok/s | First completion workload still incurred one-time JIT; do not use its P99 as steady state |
| **Local MTP 3, warmed repeat** | **201.13 tok/s** | **337.60 ms** | **903.80 ms** | **22.46 ms** | **44.5 tok/s** | Primary comparable result; 100/100 requests successful |

On the warmed repeat, local MTP 3 delivered **1.79×** the aggregate output throughput and **1.86×** the per-stream decode rate of the supplied screenshot. The screenshot's FP8 KV setting is best understood as a capacity/concurrency setting: in this local vLLM version it expanded the KV cache to 604,718 tokens, but reduced single-user text E2E speed to 52.42 tok/s and did not improve OCR latency. MTP—not FP8 KV—is the material latency/throughput accelerator in the local configuration.

## Summary: multimodal OCR / document understanding

Source workload: a 16-page Thai regulatory PDF rendered into page images. The task asks the model to identify the announcement, headings, important effective dates, and final-page notes. `OCR` here means visual-language extraction and understanding, not a character-OCR engine.

| Runtime / configuration | Image pages | Prompt tokens | Output tokens | E2E time | E2E tok/s | Prompt processing | Decode tok/s | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM, regular graph mode, BF16 | **1** | 1,148 | 489 | 40.273 s | **12.14** | — | — | Controlled first-image run on 2026-08-26; OCR completed. |
| vLLM + MTP, BF16 | **1** | 1,148 | 512 | 51.416 s | **9.96** | — | — | Controlled first-image run; OCR completed. Includes first vision/MTP kernel JIT. |
| vLLM + MTP, BF16 | **1** | 1,148 | 512 | 9.243 s | **55.39** | — | — | Same image repeated after kernels and multimodal cache were warm; not comparable to a cold document. |
| vLLM, regular graph mode, BF16 | **3 × 1** (pages 2–4) | 1,148 / request | 1,042 aggregate | 34.343 s | **30.34** | — | — | Paired warm-server, cold-image test after page-1 warm-up; per-page: 30.22 / 30.50 / 30.23 tok/s. |
| vLLM + MTP, BF16 | **3 × 1** (pages 2–4) | 1,148 / request | 1,212 aggregate | 22.213 s | **54.56** | — | — | Same paired protocol; per-page: 52.43 / 60.06 / 51.69 tok/s. |
| vLLM + MTP + FlashInfer FP8 KV (trial) | **3 × 1** (pages 2–4) | 1,148 / request | 1,080 aggregate | 20.345 s | **53.08** | — | — | Same paired protocol; per-page: 50.61 / 57.20 / 50.59 tok/s. No speed gain, and output must be quality-checked before use. |
| vLLM + MTP, BF16 | 8 | 8,127 | 677 | 14.15 s | 47.85 | — | — | Earlier run; warm image cache |
| vLLM + MTP, BF16 | 16 | 16,110 | 675 | 18.50 s | 36.48 | 1,610.8 tok/s | — | Earlier run; 50% image-cache reuse |
| llama-server UD-Q6_K_XL | 1 | 2,524 | 257 | 13.66 s | 18.82 | — | — | Summary produced but dates were confused |
| llama-server UD-Q6_K_XL | 8 | 29,079 | 309 | 36.23 s | 8.53 | 1,501.06 tok/s | 42.16 | Correct document structure and main effective date |
| llama-server UD-Q8_K_XL | 1 | 2,524 | 400 | 14.76 s | 27.11 | — | — | Hit 400-token limit |
| llama-server UD-Q8_K_XL | 8 | 29,079 | 331 | 30.23 s | 10.95 | 1,642.47 tok/s | 38.11 | Correct document number/title and main dates |
| llama-server UD-Q8_K_XL | **16** | **59,445** | **388** | **59.08 s** | **6.57** | **1,559.48 tok/s** | **36.91** | Completed full 16-page structured extraction at 64K |

The controlled one-page vLLM rows use the identical rendered first page, Thai prompt, `temperature=0`, `max_tokens=700`, 1 MP/page cap, and a 32K context. The cold runs start immediately after each server becomes healthy, so they include one-time inference JIT latency. Their response lengths differ slightly (489 vs 512 output tokens), hence E2E tok/s is the fairest compact comparison; the raw time is retained too. MTP accepted the image and produced OCR text successfully.

The **paired pages 2–4 test** is the primary runtime comparison for a new page after the service is ready: each runtime first processed page 1 only as a discarded warm-up, then received the same three previously unseen page images one at a time. MTP completed 1,212 tokens in 22.213 s (**54.56 tok/s**) versus regular vLLM's 1,042 tokens in 34.343 s (**30.34 tok/s**). MTP was therefore **1.80×** faster by aggregate E2E throughput, even though it produced 16.3% more output tokens. The different output lengths remain visible rather than being hidden by an artificial fixed-length response.

The FP8 KV trial used vLLM 0.27.1 with FlashInfer attention and `fp8_e4m3`. It reached **604,718** KV-cache tokens at the same 32K service limit, versus about 365K for the BF16-KV MTP baseline, so it is a useful capacity/concurrency option. It did **not** improve this one-user latency benchmark: text fell from 60.18 to 52.42 tok/s and paired new-page OCR from 54.56 to 53.08 tok/s. Its cold start also needs FlashInfer kernel compilation. The trial did produce OCR responses, but a representative page included questionable document-title/number inference; use BF16 KV as the accuracy-first default and validate any FP8 deployment with task-specific evaluation.

The older vLLM 8/16-page vision values are not strict apples-to-apples comparisons: their image-cache state differs, and vLLM was capped at 1 MP/page while llama.cpp received the original full-size images, so llama.cpp used substantially more image/prompt tokens.

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

1. Use **vLLM BF16 + MTP with 3 draft tokens** as the main service when interactive speed and multi-user throughput matter. Do not enable eager mode.
2. Use **llama-server UD-Q6_K_XL** for a smaller GGUF deployment: it used ~7 GB less VRAM and decoded about 16% faster than UD-Q8 in this test.
3. Use **llama-server UD-Q8_K_XL at 64K** when a single request must include 16 full-size pages. It completed the 59K-token request, but no clear OCR-quality advantage over UD-Q6 was established here.
4. For legal/regulatory workflows, use page-by-page extraction, page citations, and a second verification pass. A VLM summary is not certified character-perfect OCR.

Choose **FlashInfer + FP8 KV** only when its extra KV capacity/concurrency is needed and after a quality benchmark; it was not the fast path for a single interactive request on this setup.

For a brand-new one-page document immediately after startup, the controlled cold sample made regular vLLM faster (40.273 s vs 51.416 s), because MTP compiled speculative/vision kernels during that request. After warm-up, the paired warm-server/cold-image test showed MTP at 54.56 tok/s versus 30.34 tok/s for regular vLLM. The 55.39 tok/s same-image result remains cache-assisted and should not be treated as new-document latency.

## Cold-load time observed (WD Black 3.5-inch HDD)

| Model | Time until `llama-server` reported `model loaded` |
| --- | ---: |
| UD-Q6_K_XL | 17 min 15 s |
| UD-Q8_K_XL | 6 min 35 s |

The Ubuntu-E ext4 volume/model cache is physically stored on a **WD Black 3.5-inch mechanical HDD**. Cold-loading the 25–52 GB GGUF/BF16 model files therefore spends substantial time on mechanical-disk reads (and host page-cache state), which explains the multi-minute variance. This is a **startup I/O cost**, not serving/inference TPS. Keeping files on Ubuntu-E ext4 still avoids the additional overhead of loading from a Windows-mounted path; moving the cache to an SSD/NVMe would be the relevant way to reduce cold-load time.
