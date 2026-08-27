# PaddleOCR-VL 1.6 + PP-DocLayoutV3 + Typhoon: 24-page validation

## Outcome

For this Thai regulatory PDF, **Typhoon full-page OCR is the text reader**. It matched all 96 frozen visual anchors. PaddleOCR-VL 1.6 plus PP-DocLayoutV3 is useful for page orientation, reading order, table/region detection, and automatic crop routing, but its Markdown must **not** be merged into the Typhoon transcript automatically. It makes too many Thai-word, Thai-numeral, and table-role substitutions.

This is an image-only OCR comparison. Neither OCR model received extracted PDF text. The PDF text layer appears corrupt on some pages, so it is used only for the limited clean-prose nCER diagnostic below and can never patch a result.

## Scope and reproducible environment

| Item | Value |
| --- | --- |
| Source | 24-page Thai cloud-security standard supplied for this benchmark |
| Render | 300 DPI, exactly 2,481 × 3,508 px per page; source rotation 0° |
| Fresh input discipline | One clean directory with exactly pages 01–24; no previous crops or repair assets |
| Typhoon image input | 1,273 × 1,800 px full page |
| Typhoon model | `/root/llm-cache/typhoon-ocr1.5-2b` (4.0 GB) |
| PaddleOCR-VL model | `/root/llm-cache/PaddleOCR-VL-1.6` (1.8 GB) |
| Layout model | `/root/llm-cache/PP-DocLayoutV3` (126 MB) |
| Inference topology | Typhoon and PaddleOCR-VL: separate warm GPU vLLM services; PP-DocLayoutV3: document-parser client on CPU |
| vLLM sampling compatibility | `VLLM_USE_FLASHINFER_SAMPLER=0` only, because the local CUDA toolkit lacks `cuda_runtime.h`; multimodal attention still used FlashAttention |

The 0.20 GPU-memory reservation on each server was deliberately conservative so both readers can coexist. It is not a throughput-maximising production setting.

## Independent text and structure results

| Reader / evidence | Frozen visual anchors | Clean-prose nCER | Structure result | Appropriate use |
| --- | ---: | ---: | --- | --- |
| **Typhoon full page** | **96 / 96 = 100.0%** | **5.244%** (491 edits / 9,363 chars) | not a layout parser | **Canonical text candidate** |
| PaddleOCR-VL 1.6 + PP-DocLayoutV3 Markdown | 40 / 96 = 41.7% | 7.0704% (662 edits / 9,363 chars) | reading order/heading/list 22/24 = 91.7%; table-region association 42/47 = 89.4%; table-detection recall 47/50 = 94.0% | geometry, regions, and review routing |
| Hybrid policy | no automatic merged score | n/a | use Paddle geometry with Typhoon text, then verify disagreements on source pixels | **recommended** |

### What the metrics mean

- **Frozen anchors**: four material word/numeral/identifier checks on each of 24 rendered pages. Whitespace is ignored; any changed character or numeral fails. This is a strict field score, not whole-document CER.
- **nCER**: normalised character error rate. Only clean, non-table PDF text on pages 4–7 can be used as a reference: three pages have corrupt embedded text and 17 pages contain tables whose PDF text stream cannot represent cell geometry. It is an evaluation aid only, not input to either model.
- **Table-region association**: the detected table's content belongs to the correct visual table. It does not prove every cell is textually correct.
- **Table-detection recall**: 47 of 50 visually present source table regions were detected. Detection is valuable even when its recognised Markdown is not safe to reuse.

## Runtime observed for the fresh Typhoon pass

| Pages | Full-page size sent | Elapsed | Prompt tokens | Generated tokens | Aggregate generated throughput |
| ---: | --- | ---: | ---: | ---: | ---: |
| 24 | 1,273 × 1,800 px | 109.309 s | 59,352 | 28,477 | **260.52 tok/s** |

PaddleOCR's document-parser flow does not expose a matching token/E2E timing for its remote VLM call, so this report intentionally does not present a misleading speed comparison.

## Terra visual audit: exact text errors

The audit viewed the source PNG render and the model output. It did not use PDF-extracted text to correct a model. The table lists material, observed errors; it is not an exhaustive character diff.

### Typhoon full-page: failures outside the 96 frozen anchors

| Page | Source image | Typhoon output / structural issue | Action |
| ---: | --- | --- | --- |
| 4 | `๑๕๕` | `๑๔๕` | Verify numerical facts on image/crop |
| 8 | four-column low/medium/high matrix | rows/cells flatten or associate to the wrong column | Use layout boundaries and image review |
| 10 | `IaaS` | `laaS` | Verify identifiers/capitalisation |

### PaddleOCR-VL Markdown: material errors observed by page

| Page | Material mutations or structure issue |
| ---: | --- |
| 1 | `ไซเบอร์` → `ไซเชอร์`; gazette `เล่ม ๑๔๑ ตอนพิเศษ ๒๔๘ ง` → `เล่ม ๑๔๐ ตอนพิเศษ ๒๕๙๐ ง`; date becomes `ดอกไม้กันยายน ฝรั่งเศส`; `พ้นกำหนดสองปี` → `พันกำหนดสองปี`; invented `W.A. Leibold`. |
| 2 | `ติดตั้ง` → `ดิเด็ง`. (PaaS/SaaS/CSC/CSP anchors were exact.) |
| 3 | `๓ กันยายน ๒๕๖๗` → `6 กันยายน พ.ศ. ๒๕๖๗`; `ข้อวินิจฉัย` → `ข้อชื่อขาด`; `ไซเบอร์` → `ไซเชอร์`. |
| 4 | `ครั้งที่ ๑/๒๕๖๖` → `ครั้งที่ ๑/๒๕๖๒`; `๓๓๖` → `๓๓๖๒`; `๓๐๑` → `๓๐๑๑`; `ภัยคุกคาม` → `ภัยคุณความ`. |
| 5 | section `๑.๖` → `๑.๒`; CSP role is rendered as CSC/user; repeated `ไซเบอร์` → `ไซเชอร์`. |
| 6 | `ระดับต่ำ` → `ระดับคำ`; cited year/announcement details mutate. |
| 7 | `๓ ปี` → `ค ปี`; `คลาวด์` → `คลาด์`. |
| 8 | `ขั้นต่ำ` → `ขึ้นต่ำ`; fourth matrix header (provider certification) is labelled as user. |
| 9 | fourth matrix header (provider certification) is labelled as user; `ความมั่นคง` → `ความมันคง`. |
| 10 | third table's right header `ผู้ให้บริการคลาวด์` → `ผู้ใช้บริการคลาวด์`; role/cell association unsafe. |
| 11 | two visible tables lose their table labels and become heading/text. |
| 12 | `Cryptographic Controls` → `ntrols`; `การเข้ารหัส` → `การเข้าหัส`. |
| 13 | `ความตระหนักรู้` → `ความตระหนักรัฐ`; other security terms mutate. |
| 14 | `การบ่งชี้ข้อมูล` → `การบ่งชีวิตมูล`; further Thai word substitutions. |
| 15 | `หลายปัจจัย` → `หลายปีจัด`. |
| 16 | `๕.๒.๔.๔` → `༥.༤.༤`; `ขั้นตอน` → `ขันตอน`. |
| 17 | `Physical and Environmental Security` truncates to `Physical and security`; Thai key-management term mutates. |
| 18 | `วางแผน` → `ว่างแผน`. |
| 19 | `เข้ารหัส` → `เข้าหัวหัว`. |
| 20 | `ถูกลบ` → `ถูกเลขภาพ`; clock-synchronisation phrase is badly corrupted. |
| 21 | `ขั้นตอน` → `ชั้นตอน`; `เท่านั้น` truncates. |
| 22 | `ขอข้อมูล` → `ข้อข้อมูล`; third table's right header becomes user rather than provider. |
| 23 | `สิ้นสุด` → `สีในสุด`. |
| 24 | `ภัยคุกคาม` → `ภัยคุณความ`; second table's right header becomes user rather than provider; `เกี่ยวกับ` → `เกียวกับ`. |

Paddle reading order/heading/list failed on pages 1 (hallucinated `W.A. Leibold`) and 16 (Tibetan-like section numerals). The five detected-table association failures are pages 8, 9, 10, 22, and 24. A "user" versus "provider" column swap is a semantic error, even if the text itself is recognisable.

## Safe automatic hybrid design

This is deliberately **auto-detect, not fixed-bbox**:

1. Render the original PDF page at 300 DPI and retain its page number and image hash.
2. Ask PP-DocLayoutV3 for orientation confidence, text/table/heading regions, reading order, and table boxes. Auto-rotate only at high confidence with a clear top-two margin; otherwise leave the source image unchanged for review.
3. Send the complete orientation-corrected page to Typhoon for the primary transcript. Keep its source image and request metadata.
4. Use the dynamically detected Paddle regions only to queue targeted review crops: tables, numerals/dates, legal citations, identifiers, headings, and low-confidence/overlap regions. Record the derived coordinate, detection label, and source hash for each crop.
5. Do **not** concatenate or replace Typhoon text with Paddle Markdown. For a detected table, preserve Paddle's box/cell-tree evidence and Typhoon's textual candidate independently.
6. If outputs disagree, or a value is a number, legal date, identifier (`IaaS`/`PaaS`/`SaaS`), or left/right role, require source-image review. Only an explicit accepted field-level repair can replace that field.

This retains the practical strength of PP-DocLayoutV3 without treating a layout model's recognised text as a trusted correction source.

## Commands used

Download the two Paddle models into Ubuntu-E (resumable Xet downloads):

```bash
/opt/vllm/bin/python scripts/download_paddle_models.py --cache-root /root/llm-cache
```

Start the two GPU model services; the sampler setting is the local build workaround described above, not a general performance recommendation:

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_PLUGINS='' VLLM_USE_FLASHINFER_SAMPLER=0 \
  /opt/vllm/bin/vllm serve /root/llm-cache/typhoon-ocr1.5-2b \
  --served-model-name typhoon-ocr-1-5 --host 127.0.0.1 --port 8091 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.20

VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_PLUGINS='' VLLM_USE_FLASHINFER_SAMPLER=0 \
  /opt/vllm/bin/vllm serve /root/llm-cache/PaddleOCR-VL-1.6 \
  --served-model-name paddleocr-vl-1-6 --host 127.0.0.1 --port 8092 \
  --max-model-len 16384 --max-num-seqs 1 --gpu-memory-utilization 0.20 \
  --trust-remote-code
```

Run layout-guided document parsing against exact page PNGs:

```bash
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
  /root/venvs/ppstructurev3/bin/paddleocr doc_parser \
  -i /path/to/clean-24-page-pngs --pipeline_version v1.6 \
  --layout_detection_model_dir /root/llm-cache/PP-DocLayoutV3 \
  --vl_rec_backend vllm-server --vl_rec_server_url http://127.0.0.1:8092/v1 \
  --vl_rec_api_model_name paddleocr-vl-1-6 --vl_rec_max_concurrency 1 \
  --use_doc_orientation_classify false --use_doc_unwarping false \
  --format_block_content true --max_new_tokens 4096 --device cpu \
  --save_path /path/to/paddle-vl-output
```

Evaluate only the clean prose subset after the OCR run:

```bash
/opt/vllm/bin/python scripts/evaluate_paddleocr_vl_output.py \
  --pdf '/mnt/e/project/[สกมช] มาตรฐานด้านการรักษาความมั่นคงปลอดภัยไซเบอร์ระบบคลาวด์ พ.ศ. ๒๕๖๗.pdf' \
  --results-dir /path/to/paddle-vl-output --output /path/to/paddle-ncer.json
```

The generated OCR text, rendered pages, and evaluation JSON are intentionally not committed: they derive from the supplied source document and are retained only in the local test workspace.
