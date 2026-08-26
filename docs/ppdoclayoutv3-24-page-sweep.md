# PP-DocLayoutV3 + Qwen OCR — 24-page resolution and crop sweep

Date: 2026-08-26. This is an accuracy/latency comparison on one RTX PRO 6000 Blackwell Workstation Edition (96 GB), using the local Qwen3.8-27B vLLM service with BF16 weights, BF16 KV (`auto`), MTP 3, `max-model-len=81920`, and `max_pixels=3998400` as the server ceiling. Model loading is excluded.

The source was the 24-page Thai cloud-security standard PDF supplied for this benchmark. It is **not** committed to this repository. Each page was rendered at 300 DPI, **2481 × 3508 px**. The PDF text layer was not used as ground truth because its Thai extraction is unreliable on some pages.

## Result first

**Primary OCR record: full-page 2000 px.** It has the best complete-page source-verified result: **92/96 = 95.8%** across four visual anchors per page, at **50.30 E2E output tok/s** and **11.25 s/page**.

Do not replace it with the apparently higher-scoring PP-DocLayoutV3 crop. The 2200-px crop reaches **93/96 = 96.9%**, but intentionally removed the page-1 official gazette metadata and large title/issuer block. It is a useful **second pass for prose clauses/dates**, not an archival full-page OCR result.

| Use case | Input | Source-verified result | Performance | Recommendation |
| --- | --- | ---: | ---: | --- |
| Primary transcript / search record | Full page, 2000 profile | **92/96 (95.8%)** | **50.30 tok/s**, **11.25 s/page** | **Use this** |
| Faster complete-page draft | Full page, 1800 profile | 90/96 (93.8%) | 50.34 tok/s, 11.24 s/page | Use if the small quality loss is acceptable |
| Extra check of prose/date clause | PP-DocLayoutV3 crop, 2200 profile | 93/96 (96.9%), **not complete page** | 51.37 tok/s, 10.85 s/page | Second pass only; retain full page |
| 1400 px | Full page | Not valid for full-document comparison: pages 17–24 hit output cap | 57.01 tok/s, 12.86 s/page | Do not use this batching/output limit |

## 2600 px pagewise accuracy-control

This is deliberately a different protocol from the eight-page sweep: **one 1824 × 2592 image and one HTTP request at a time**, repeated for all 24 pages after a discarded page-1 warm-up. It used Qwen MTP 3, `kv-cache-dtype=auto`, `max-model-len=32768`, `max-num-seqs=1`, `gpu-memory-utilization=0.80`, and `--enforce-eager`. Eager mode was selected only because CUDA-graph profiling in this WSL driver environment failed during a 1.43-GB allocation; its speed must not be compared as a production graph-mode speed result.

| Input | Requests | Prompt / completion tokens | E2E elapsed | E2E output tok/s | Seconds/page | Finish status | Terra anchors |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Full 2600 (1824 × 2592; 4.728 MP) | 24 × 1 page | 113,391 / 13,538 | 390.79 s | 34.64 | 16.28 | All 24 `stop` | 76/96 = 79.2% |

The score is **not** a full-text CER; it is four source-visual anchors per page. Nevertheless it is materially below full 2000 (92/96 = 95.8%), while also being slower. The meaningful errors are structural and legal/numeric rather than output truncation: page 1 changes `๒๔๘` to `๒๘๘` and replaces the two-year effective clause; page 3 changes 3 September to 30 September; page 4 changes `๖๓๒/๑๕๕/๑๔๘` and threat numbers; pages 5–8 alter section numbering/table structure; and page 22 changes `๕.๒.๙` family numbering. Therefore **do not increase beyond the 2000 full-page profile as an OCR-quality remedy**.

### Approved hybrid patches only

PP-DocLayoutV3 supplies region geometry; a field crop is then read by Qwen and visually audited. A crop never replaces the page transcript. For the 2600 control, Terra approved only this allowlist:

| Page | Crop field | Approved replacement | Do not use |
| ---: | --- | --- | --- |
| 3 | Signature/date line | `ประกาศ ณ วันที่ ๓ กันยายน พ.ศ. ๒๕๖๗`; `ภูมิธรรม เวชยชัย` | Any other page-3 crop text |
| 4 | Education statistic | `๖๓๒` | The rest of the statistics paragraph |
| 4 | Appendix/date crop | token `แนบท้ายประกาศ`; date `๒๒ ธันวาคม ๒๕๖๖` | Treating the crop as the full title/transcript |

The dedicated gazette crop misread `๑๔๑` as `๑๘๑`; the effective-date crop still wrote the following-day rule; other statistic and threat crops produced wrong numbers or invented values. Those outputs are explicitly rejected. The PDF text layer was also tested and has mojibake on pages 1 and 3, so it is not used as ground truth or as a hybrid patch source.

### PaddleOCR status on this host

PP-DocLayoutV3 is installed and completed layout detection for all 24 pages; its boxes are the crop source used above. The base Thai PaddleOCR detector/recognizer weights were also downloaded, but **no PaddleOCR transcript is included in the scores**: a normal CPU run failed in Paddle's oneDNN/PIR path, and the fallback attempted to allocate about 31.5 GiB of WSL memory before the Linux process was OOM-killed. A smaller 2600-px retry also made the Ubuntu-E WSL service unstable. PaddleOCR-VL 1.6 is **not installed yet**; the required GPU Paddle wheel is a 1.8-GB download and remains incomplete. Until that GPU environment installs and produces independently checked output, full-page Qwen 2000 plus PP-DocLayoutV3 field crops remains the reproducible accuracy recommendation.

## One-page full + PP-DocLayoutV3 detail repair trial

This is the stricter reusable workflow: send the **2000 full page and an original-resolution detail crop together in one vLLM MTP 3 request**, then ask for one named field only. PP-DocLayoutV3 supplies only crop geometry; it does not provide untrusted OCR text to Qwen. One full-page-plus-detail warm-up was discarded. Codex Terra checked the four requests below character-for-character against the 300-DPI render.

| Source field | PP-DocLayoutV3-derived detail | Full + detail prompt result | E2E / prompt tokens | Terra strict result | Field patch decision |
| --- | --- | --- | ---: | --- | --- |
| Page 1 gazette header | One `header` box, padded: `(209,269,1026,466)` | `เล่ม ๑๔๑ ตอนพิเศษ ๒๘๘ ง` | 1.37 s / 2,995 | **Wrong**: `๒๔๘` → `๒๘๘` | Reject |
| Page 1 clause 2 | One `text` box, padded: `(239,1597,2233,1804)` | `ข้อ ๒ ประกาศนี้ให้ใช้บังคับเมื่อพ้นกำหนดสองปีนับแต่วันประกาศในราชกิจจานุเบกษาเป็นต้นไป` | 1.80 s / 3,224 | **Exact** | **Approve this field only** |
| Page 3 signing role | One `text` box, padded: `(1074,1109,1821,1316)` | `รองนายกรัฏมนตรี  ปลื้บิบัติหน้าที่` | 1.62 s / 2,980 | **Wrong**: `รัฐ` → `รัฏ`; `ปฏิบัติ` mutates | Reject |
| Page 4 meeting field | First `text` box/body tile: `(300,740,2283,1040)` | `เมื่่อวันที่ ๒๒ ธันวาคม ๒๕๖๖ ณ ตึกบัญชาการ ๑ ทำเนียบรัฐบาล` | 1.82 s / 3,401 | **Wrong under exact OCR**: duplicated tone mark in `เมื่่อ`; all other target words/numerals match | Reject |

Strict score: **1/4 = 25%**. The approved page-1 clause replaces only that span in the full-page transcript; if that source-reviewed replacement is applied, the original 92/96 anchor record becomes **93/96 = 96.9%**. This is a field-level result, not a new whole-document OCR/CER score.

Two narrower retries were also rejected: the page-3 role-line crop produced `รองนายกรรฐมนตรี ปลื้บตติหน่าที่`; the page-4 location-only request hallucinated a different meeting room. Crop detail is therefore a useful **verification-candidate generator**, not a safe automatic repair mechanism. In this document it repaired the legal effective clause but did not reliably repair small header digits, Thai job titles, or the entire meeting metadata.

### Paddle orientation preflight: required before layout

Use PaddleOCR's small `PP-LCNet_x1_0_doc_ori` classifier before PP-DocLayoutV3. Its label represents the corrective counter-clockwise rotation. A controlled calibration used page 1 rotated in advance: input CCW 90° was classified as correction `270°` (0.921), input CW 90° as `90°` (0.928), and input 180° as `180°` (0.918).

On the 24 original 300-DPI pages, pages 1–19 and 21–24 classified as upright `0°` with 0.899–0.932 confidence and 0.84–0.91 top-two margin. Page 20 is visually upright but classified as `180°` at only 0.504 confidence with a 0.067 margin over `0°`; it was manually approved as `0°` and **not rotated**. The standard gate is confidence ≥ 0.80 **and** top-two margin ≥ 0.20. Anything else requires source-image review and an explicit recorded override.

```bash
# 1. Classify orientation; it is separate from full Paddle OCR.
/root/venvs/ppstructurev3/bin/paddleocr doc_img_orientation_classification \
  -i /path/to/rendered-pages --save_path /path/to/orientation \
  --topk 4 --device cpu --enable_mkldnn False --cpu_threads 2

# 2. Apply only an accepted angle. Page 20 below is a source-reviewed override,
# not a blind rotation.
/root/venvs/ppstructurev3/bin/python scripts/apply_paddle_orientation.py \
  --input-dir /path/to/rendered-pages --orientation-dir /path/to/orientation \
  --output-dir /path/to/upright-pages --approved-orientation 20=0

# 3. Only then detect layout and build detail crops from the upright images.
/root/venvs/ppstructurev3/bin/python scripts/prepare_layout_ocr_images.py \
  --input-dir /path/to/upright-pages --output-dir /path/to/layout \
  --model-dir /root/llm-cache/pp-doclayoutv3-safetensors --device cpu \
  --profile 2000=1408x1984
```

## What “1400 / 1800 / 2000 / 2200” means here

These are actual image inputs, not an enlarged `max_model_len` reservation. To keep Qwen's vision grid deterministic, the full A4 inputs were pre-resized to these exact 32-pixel-grid dimensions before sending them to the already-warm high-resolution server:

| Profile name | Full-page input dimensions | Image area | Input-token observation, 24 pages |
| --- | ---: | ---: | ---: |
| 1400 | 960 × 1376 | 1.321 MP | 31,515 prompt tokens |
| 1800 | 1280 × 1792 | 2.294 MP | 54,315 prompt tokens |
| 2000 | 1408 × 1984 | 2.793 MP | 66,027 prompt tokens |
| 2200 | 1536 × 2176 | 3.342 MP | 78,891 prompt tokens |

The vLLM server's `max_pixels=3998400` (about a 2400-px A4 ceiling) remains above all these inputs, so it does not downsize them further. This experiment therefore isolates the delivered image resolution while holding weights, MTP depth, KV dtype, context ceiling, prompt, batching, and sampling fixed.

## Full-page benchmark, all 24 pages

Each profile uses three fresh HTTP requests of eight pages, `temperature=0`, Qwen thinking disabled, and a budget of 1,024 output tokens per page (8,192 max per request). Each run used cache-distinct image bytes. `E2E` begins when the client sends the request and ends when the complete response arrives; it includes image preprocessing, visual prefill, decoding, and API overhead, but not model startup.

| Full-page profile | Prompt tokens | Output tokens | E2E elapsed | E2E output tok/s | Seconds/page | Finish status | Terra anchor score |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1400 | 31,515 | 17,599 | 308.70 s | **57.01** | 12.86 | Batch 17–24 hit `length`; invalid full-doc comparison | — |
| 1800 | 54,315 | 13,574 | 269.65 s | 50.34 | **11.24** | All three requests `stop` | 90/96 = 93.8% |
| **2000** | **66,027** | **13,583** | **270.05 s** | **50.30** | **11.25** | All three requests `stop` | **92/96 = 95.8%** |
| 2200 | 78,891 | 13,662 | 276.38 s | 49.43 | 11.52 | All three requests `stop` | 91/96 = 94.8% |

The 1800- and 2000-px profiles have virtually the same observed latency, while 2000 adds two verified anchors. 2200 spends more image tokens and time without improving the complete-page score. The fast 1400 output-rate is misleading: its final eight-page response stopped at the 8,192-token limit, repeats text on page 24, and ends mid-sentence.

### Every page checked by Codex Terra

Terra visually compared each OCR response with the 300-DPI render. The rubric is **four page-specific anchors per page** (96 total), with whitespace normalized. An anchor is exact (`E`), wrong substitution (`W`), omitted (`O`), or hallucinated (`H`). It is deliberately **not a character-error-rate (CER) claim** and must not be read as document-perfect transcription.

| Page | 1400 full | 1800 full | 2000 full | 2200 full | Anchor family / material observation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2/4 | 2/4 | 2/4 | 2/4 | Gazette issue and effective-after-two-years clause are wrong at all sizes |
| 2 | 4/4 | 4/4 | 4/4 | 4/4 | PaaS/SaaS, CSC/CSP, 30-day report |
| 3 | 2/4 | 3/4 | 3/4 | 2/4 | Source date is 3 Sep 2567; several outputs say 30 Sep |
| 4 | 1/4 | 2/4 | 3/4 | 3/4 | Appendix, meeting date/location, and statistics are fragile |
| 5 | 4/4 | 4/4 | 4/4 | 4/4 | Governance sections / English standards |
| 6 | 4/4 | 4/4 | 4/4 | 4/4 | Certification and assessment cycle |
| 7 | 4/4 | 4/4 | 4/4 | 4/4 | ISO/IEC / CSA STAR / normative reference |
| 8 | 4/4 | 3/4 | 4/4 | 4/4 | Four-column matrix retained, but headers/lists can mutate |
| 9 | 4/4 | 4/4 | 4/4 | 4/4 | Policy table and two roles |
| 10 | 4/4 | 4/4 | 4/4 | 4/4 | Organization/authority contacts; blank cell preserved |
| 11 | 4/4 | 4/4 | 4/4 | 4/4 | Compliance, legislation, IP, records |
| 12 | 4/4 | 4/4 | 4/4 | 4/4 | Cryptographic controls and independent review |
| 13 | 4/4 | 4/4 | 4/4 | 4/4 | Infrastructure / HR / asset management |
| 14 | 4/4 | 4/4 | 4/4 | 4/4 | Labelling/access controls; genuine blank cell preserved |
| 15 | 4/4 | 4/4 | 4/4 | 4/4 | Privileged access/authentication/utilities |
| 16 | 3/4 | 4/4 | 4/4 | 4/4 | Secure log-on and key management |
| 17 | 4/4* | 4/4 | 4/4 | 4/4 | Key management / physical security |
| 18 | 4/4* | 4/4 | 4/4 | 4/4 | Operations/change/capacity/backup |
| 19 | 4/4* | 4/4 | 4/4 | 4/4 | Backup/logging/retention |
| 20 | 4/4* | 4/4 | 4/4 | 4/4 | Operator logs/clock/vulnerabilities |
| 21 | 4/4* | 4/4 | 4/4 | 4/4 | Transfer/network/SDLC |
| 22 | 4/4* | 4/4 | 4/4 | 4/4 | Supplier agreements / six bullets |
| 23 | 4/4* | 4/4 | 4/4 | 4/4 | Supply chain / incident management |
| 24 | 1/4* | 4/4 | 4/4 | 4/4 | 1400 repeats/hallucinates due output cap |

\* The 1400 final request is length-capped, so its apparent page 17–23 matches cannot establish a valid end-to-end document result.

Full-page status totals: 1800 = `E90/W6/O0/H0`; 2000 = `E92/W4/O0/H0`; 2200 = `E91/W4/O1/H0`.

### Errors that remain material at 2000 px

The high score does not permit unverified regulatory transcription. In particular:

| Page | Source visual fact | Observed issue |
| ---: | --- | --- |
| 1 | Gazette issue `๒๔๘`; effectiveness is **after two years** | All profiles mutate the issue and write the effective rule as the following day |
| 3 | Date: 3 September 2567 | 1800 and 2200 change it to 30 September; commission wording also mutates in some profiles |
| 4 | Appendix; meeting 22 Dec 2566; statistics include 632 / 155 / 148 and 515 / 336 / 301 | 2000 preserves the checked date/first values but still changes location, some statistics, and a threat term |
| 8–24 | Visually structured role tables | The model creates readable ordered text, not geometry-faithful cells, CSV, or a safe cell-by-cell record |

For legal titles, dates, gazette issue/section numbers, locations, numeric statistics, and every material table cell: retain the page image and run a focused crop/verification pass.

## PP-DocLayoutV3 crop A/B

PP-DocLayoutV3 was run on the original 300-DPI page to detect semantic regions. The `layout-content` variant takes the padded union of detected non-header/non-footer/non-page-number regions, then resizes **without changing aspect ratio**. It keeps image area at or below the matching full-page profile. This was a real pipeline run over all 24 pages; its JSON manifest records all boxes and crop rectangles.

| Variant | Prompt tokens | Output tokens | E2E elapsed | E2E output tok/s | Seconds/page | Terra score | Complete-page safe? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full 1800 | 54,315 | 13,574 | 269.65 s | 50.34 | 11.24 | 90/96 = 93.8% | Yes |
| Crop 1800 | 48,769 | 13,265 | 244.49 s | **54.25** | **10.19** | 90/96 = 93.8% | **No** — title removed |
| Full 2200 | 78,891 | 13,662 | 276.38 s | 49.43 | 11.52 | 91/96 = 94.8% | Yes |
| Crop 2200 | 71,466 | 13,379 | 260.43 s | 51.37 | 10.85 | **93/96 = 96.9%** | **No** — title/gazette metadata removed |

Crop-page scores (1800 / 2200) were: p1 2/4 / 3/4, p2 4/4 / 4/4, p3 3/4 / 3/4, p4 2/4 / 3/4, p5–7 4/4 / 4/4, p8 3/4 / 4/4, p9–24 4/4 / 4/4. Status totals: crop-1800 = `E90/W5/O1/H0`; crop-2200 = `E93/W2/O1/H0`.

The one crop omission is material: on page 1 the crop physically removes the official title/issuer and gazette metadata. It also does not make header policy reliable: page 3 still emits a wrong gazette header. Therefore:

1. Keep **full-page 2000** as the main OCR record.
2. Use **crop 2200** only as a second pass for a prose clause, date, or low-confidence text.
3. For page 1/3 metadata, make a dedicated header crop; do not rely on automatic margin trimming.
4. For pages 8–24, crop a **specific visual table row/cell** when an exact role-to-requirement association matters.

## Installation and reproducible run

PP-DocLayoutV3 is used for layout geometry, not Thai character recognition. Qwen remains the VLM OCR engine. The detector runs CPU-only in a separate Ubuntu-E environment, so it does not contend with the vLLM GPU service.

```bash
# Ubuntu-E: isolated CPU environment
uv venv --python /usr/bin/python3 /root/venvs/ppstructurev3
uv pip install --python /root/venvs/ppstructurev3/bin/python \
  paddlepaddle 'paddleocr[doc-parser]' transformers
uv pip install --python /root/venvs/ppstructurev3/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Public 133-MB Transformer/safetensors detector, persistent local cache
/root/venvs/ppstructurev3/bin/hf download \
  PaddlePaddle/PP-DocLayoutV3_safetensors \
  --local-dir /root/llm-cache/pp-doclayoutv3-safetensors

# Build overlay, crop manifest, and all Qwen-grid input variants.
/root/venvs/ppstructurev3/bin/python scripts/prepare_layout_ocr_images.py \
  --input-dir /path/to/rendered-300dpi-pages \
  --output-dir /path/to/ocr-ppdoclayoutv3 \
  --model-dir /root/llm-cache/pp-doclayoutv3-safetensors \
  --device cpu \
  --profile 1400=960x1376 \
  --profile 1800=1280x1792 \
  --profile 2000=1408x1984 \
  --profile 2200=1536x2176

# Benchmark 24 full pages as three fresh eight-page requests.
/root/venvs/ppstructurev3/bin/python scripts/benchmark_vllm_ocr_modes.py \
  --mode sharded-batch --batch-size 8 \
  --images /path/to/ocr-ppdoclayoutv3/images/2000/full/page-*.png \
  --max-tokens-per-page 1024 --cache-buster-id 600 \
  --output /path/to/results/2000-full.json
```

### Automatic detail candidates — no fixed bounding boxes

The prior four-field repair experiment used source-reviewed rectangles to
isolate known errors.  Those rectangles are retained only as historical
evaluation evidence; they are **not** a reusable OCR policy and must not be
copied to a different document.  A production repair path needs to discover
every crop from the document itself.

[`build_auto_layout_ocr_candidates.py`](../scripts/build_auto_layout_ocr_candidates.py)
does that from PP-DocLayoutV3's `layout-manifest.json`.  It reconstructs
header/footer/page-number lines from vertical alignment, emits every detected
semantic block, and splits only a *detected* over-height text block into
overlapping tiles.  Padding is a fraction of that region's own geometry—not a
page coordinate.  The output manifest records the original detector boxes,
the resulting crop, reading order, source hash, and the image dimensions sent
to the OCR reader.

```bash
# Generates detail evidence for every detected region; no --crop X,Y,R,B input.
/root/venvs/ppstructurev3/bin/python scripts/build_auto_layout_ocr_candidates.py \
  --layout-manifest /path/to/ocr-ppdoclayoutv3/layout-manifest.json \
  --output-dir /path/to/ocr-auto-candidates \
  --model-max-side 1800
```

### Typhoon OCR as independent evidence

Typhoon OCR 1.5-2B is installed persistently in Ubuntu-E at
`/root/llm-cache/typhoon-ocr1.5-2b` (4.0 GB, BF16).  It is a separate
Qwen3-VL 2B OCR model, so it can read the automatic crops as an independent
second opinion rather than asking Qwen3.8 to validate its own text.  Its model
card is prompt-specific: use the supplied OCR prompt verbatim, `temperature=0`,
and its 1,800-px image policy.  Do not repurpose it for arbitrary
question-answering over a crop.

```bash
# A deliberately small VRAM reservation for Typhoon while Qwen is stopped.
# The official card documents a 49,152-token configuration; 16K is sufficient
# for one 1,800-px detail candidate and leaves the RTX PRO 6000 available.
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_PLUGINS='' \
  /opt/vllm/bin/vllm serve /root/llm-cache/typhoon-ocr1.5-2b \
  --served-model-name typhoon-ocr-1-5 --host 127.0.0.1 --port 8091 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.20

# The runner reads Typhoon's supplied prompt from its local README, and writes
# an evidence JSON.  Omit the filters to process every automatic candidate.
/opt/vllm/bin/python scripts/run_typhoon_candidate_ocr.py \
  --candidate-manifest /path/to/ocr-auto-candidates/candidate-manifest.json \
  --typhoon-model-readme /root/llm-cache/typhoon-ocr1.5-2b/README.md \
  --page 1 --candidate-id header_line-001 \
  --output /path/to/results/typhoon-p1-header.json
```

#### Warm-server four-field validation — automatic crops

The candidate builder was run over all 24 source pages and made **165**
detail crops.  The following four were selected only because the previous
full-page Qwen run had known strict-anchor errors.  They were generated from
their detected semantic boxes/line grouping; the table intentionally gives
candidate identity and image dimensions rather than reusable page coordinates.
Typhoon used the prompt parsed from its model README at `temperature=0`.
Codex Terra then compared each reported anchor to the original rendered page.

| Page | Automatic candidate | Detected evidence | Typhoon image sent | Warm E2E | Terra strict result | Scope |
| ---: | --- | --- | ---: | ---: | --- | --- |
| 1 | `header_line-001` | Vertically aligned `header` fragments grouped into one line | 1800 × 155 | 0.275 s | **Accept** — gazette issue `๒๔๘` exact | This header field only |
| 1 | `text-005` | One detected text region, reading order 5 | 1800 × 155 | 0.190 s | **Accept** — effective after two years | This clause only |
| 3 | `text-005` | One detected text region, reading order 5 | 1800 × 264 | 0.099 s | **Accept** — signing role exact | This role only |
| 4 | `text-001` | One detected text region, reading order 4 | 1800 × 621 | 1.330 s | **Accept** — meeting date/location exact | Those meeting fields only |

This is **4/4 strict anchors**, not 100% character or whole-page accuracy.
The page-4 candidate contains a larger paragraph than its target sentence, so
only the source-checked meeting fields are eligible as replacements.  No
result from the remaining 161 automatic candidates has been promoted without
independent validation.

The safe automatic repair decision is deliberately conservative:

1. Paddle orientation classification rotates only high-confidence pages;
   ambiguous pages retain their pixels and are queued for review.
2. PP-DocLayoutV3 detects regions and the automatic candidate builder makes
   crops without fixed coordinates.
3. Qwen full-page 2000 remains the canonical OCR record.  Typhoon reads only
   the candidate crop using its documented prompt.
4. A repair is accepted only with an independently checkable match (or
   explicit source-image/Terra approval).  A disagreement is a review item,
   never an automatic replacement.  PDF embedded text is auxiliary evidence
   only when a per-region text-health check says it is clean; corrupted Thai
   text is discarded.

[`apply_paddle_orientation.py`](../scripts/apply_paddle_orientation.py) converts Paddle orientation classifications into upright images plus an auditable rotation manifest, refusing low-confidence automatic rotations. [`prepare_layout_ocr_images.py`](../scripts/prepare_layout_ocr_images.py) then saves a `layout-manifest.json`, overlays, full-page images, and crop images. [`benchmark_vllm_ocr_modes.py`](../scripts/benchmark_vllm_ocr_modes.py) records per-request response text, token counts, finish reason, and E2E time; its `--detail-image` option adds a crop as an image beside one full page without injecting unverified crop text. `sharded-batch` prevents a 24-image request from being conflated with a single context-length experiment.

The relevant upstream references are [PP-DocLayoutV3 safetensors](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors/tree/main), [PaddleOCR layout detection](https://www.paddleocr.ai/latest/en/version3.x/module_usage/layout_detection.html), [PP-StructureV3](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html), and [Typhoon OCR 1.5-2B](https://huggingface.co/typhoon-ai/typhoon-ocr1.5-2b).
