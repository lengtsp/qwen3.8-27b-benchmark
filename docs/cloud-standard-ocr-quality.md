# Cloud-security standard OCR quality rubric

This is a manual, visual **field-accuracy** check for the OCR benchmark in the root README. It is not character error rate (CER) and it is not a claim about every character of the 24-page document. The source PDF and full model responses are deliberately not committed. Review used the rendered page images as the ground truth because the PDF's embedded Thai text layer is corrupt on some pages.

## Scoring rule

- One field is worth one point.
- A field passes only when the value at its intended location is correct.
- A correct word appearing elsewhere does **not** rescue a field if the title/date/number at the intended location was changed. This is deliberately strict because the document is regulatory.
- `Incorrect` includes omission, mutation, or invented replacement text.

The directly comparable total has 21 fields: 10 from page 1 processed alone, plus 11 from page 4 in the eight-page batch. Both runs use MTP 3, BF16 weights, 32K context, 1,048,576 effective image pixels/page and a 700-token/page cap. The only comparison variable is BF16 versus FP8 E4M3 KV cache.

## Page 1, one-page OCR: 5 / 10 correct (50.0%) for both profiles

| Visual checkpoint | Expected value in image | MTP 3 BF16 | MTP 3 FP8 KV |
| --- | --- | --- | --- |
| Printed page number | `หน้า ๑๖` | Correct | Correct |
| Gazette issue | `เล่ม ๑๔๑ ตอนพิเศษ ๒๔๘ ง` | Incorrect: `๒๘ ง` | Incorrect: `๒๘ ง` |
| Gazette date | `๑๐ กันยายน ๒๕๖๗` | Correct | Correct |
| Issuer | Cyber Security Committee announcement | Correct | Correct |
| Main title | `...ระบบคลาวด์` | Incorrect: `...ระดับคลาวด์` | Incorrect: `...ระดับชาติ` |
| Standard year | `พ.ศ. ๒๕๖๗` | Correct | Correct |
| Opening legal citation | `พระราชบัญญัติการรักษาความมั่นคงปลอดภัยไซเบอร์ พ.ศ. ๒๕๖๒` | Incorrect: mutated `พระราชนิพนธ์บัญญัติ` | Incorrect: mutated opening citation |
| Committee meeting | `ครั้งที่ ๒/๒๕๖๗`, `๓๑ กรกฎาคม ๒๕๖๗` | Incorrect: date changed to `๓๐ กรกฎาคม ๒๕๖๗` | Incorrect: date changed to `๓๐ กรกฎาคม ๒๕๖๗` |
| Effective-date rule | Effective **after two years** from gazette publication | Incorrect: changed to the following day | Incorrect: changed to the following day |
| Cloud Computing definition | `การประมวลผลคลาวด์` / `Cloud Computing` | Correct | Correct |

## Page 4 inside the eight-page batch: 7 / 11 correct (63.6%) for both profiles

| Visual checkpoint | Expected value in image | MTP 3 BF16 | MTP 3 FP8 KV |
| --- | --- | --- | --- |
| Appendix heading | `แนบท้ายประกาศคณะกรรมการ...` | Incorrect: changed to `แนวทาง...` | Incorrect: changed to `แนวทาง...` |
| Main title at heading | `...ระบบคลาวด์` | Incorrect: changed to `...ระดับคลาวด์` | Incorrect: changed to `...ระดับคลาวด์` |
| Standard year | `พ.ศ. ๒๕๖๗` | Correct | Correct |
| Section | `๑. บทนำ` | Correct | Correct |
| Subsection | `๑.๑ เหตุผลความจำเป็น` | Correct | Correct |
| Meeting number | `ครั้งที่ ๑/๒๕๖๖` | Correct | Correct |
| Meeting date | `๒๒ ธันวาคม ๒๕๖๖` | Incorrect: `๒๖ มีนาคม ๒๕๖๖` | Incorrect: `๒๖ มีนาคม ๒๕๖๖` |
| Location | `ตึกบัญชาการ ๑` | Correct | Correct |
| Policy name | `Cloud First Policy` | Correct | Correct |
| Duration | `ระยะ ๕ ปี` | Correct | Correct |
| Cyber-incident figure | `๖๓๒ ครั้ง` | Incorrect: `๖๒๒ ครั้ง` | Incorrect: `๖๒๒ ครั้ง` |

## Comparable total

| Profile | Correct fields | Incorrect fields | Strict field accuracy | Field error / mutation rate |
| --- | ---: | ---: | ---: | ---: |
| MTP 3 + BF16 KV | 12 / 21 | 9 / 21 | **57.1%** | **42.9%** |
| MTP 3 + FP8 E4M3 KV | 12 / 21 | 9 / 21 | **57.1%** | **42.9%** |

These results were independently re-checked visually by Codex Terra against the PNG renders. The earlier 14/21 count mistakenly credited the page-1 meeting date and effective-date rule. In the image, the meeting date is **31 July 2567**, not 30 July, and clause 2 makes the standard effective **after two years**, not on the following day.

The `Cloud First Policy` row above awards the literal English term. If the checkpoint instead requires the complete Thai-and-English policy name, BF16 stays 12/21 while FP8 is **11/21 (52.4%)** because its Thai wording is changed. An additional non-comparable FP8 page-4-only probe scored 6/11 (54.5%) and also altered the title, meeting date and statistics. It is not included in the 21-field total because there is no matching BF16 page-4-only run.

## Interpretation

At 1 MP effective resolution, Qwen3.8-27B is useful for locating sections, headings, familiar English terms, and a first-pass Thai transcription. It is not reliable for exact legal titles, government dates, issue numbers, legal effective dates, or numerical claims in this source. The 21-field score is intentionally only an anchor-field metric, not a full-page score: both outputs also altered unscored page-4 statistics, the `1.3 ฐานอำนาจ` heading, and the cited legal authority. It is **not CER**; calculating CER would require complete normalized ground truth plus alignment, while these responses are capped and can be truncated. The benchmark therefore recommends MTP 3 + BF16 KV for speed/quality between these two vLLM profiles, with mandatory image/PDF verification of every final regulatory value.
