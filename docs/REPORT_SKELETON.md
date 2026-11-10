# DocuLens — report skeleton

This is the **hand-written outline** for the written report: what each section must
contain and which generated asset backs it.

`doculens report` writes the **generated counterpart** to `results/REPORT.md`, with
the same section order and the real numbers from the last run already substituted.
This file is never overwritten, so the outline stays under version control while the
numbers stay reproducible. Write the report by taking `results/REPORT.md` and
expanding the prose; do not retype numbers by hand.

---

## 1. Problem statement

- Indian insurance claim intake receives claim forms and garage invoices as phone
  photos and rescans.
- Capture defects: defocus, sensor noise, skew, uneven lighting, recompression.
- OCR error propagates into every downstream step (field extraction, policy
  validation, settlement).
- **The measurable question:** how much recognition loss can classical DIP recover
  before the OCR engine sees the page?

## 2. Data

- Synthetic corpus: document count, two document types, canvas size, Faker `en_IN`.
- Why synthetic: exact ground truth, which is what makes CER/WER meaningful; real
  claim documents cannot be published with a course report.
- Degradations and their three severity levels; the `none` condition.
- Train/val/test split, **fixed before any tuning**; only `test` is reported.
- *Assets:* `data/manifest_clean.csv`, `data/manifest_degraded.csv`, a sample page.

## 3. Method

- Block diagram: Input → DIP preprocessing → Segmentation → OCR → Evaluation, with
  the dashed future branch (Extraction → RAG → Agent → MCP).
- The stage table for the headline pipeline, with parameters.
- Why each stage is there, in one line each.
- Metric definitions and the exact normalisation applied before scoring.
- *Assets:* `results/figures/block_diagram.png`, `results/figures/stages_*.png`,
  `results/figures/segmentation_regions.png`.

## 4. Results

- Baseline vs pipelines: overall mean CER/WER per method.
- Mean CER by degradation and method.
- CER against severity level.
- *Assets:* `results/tables.md`, `results/figures/cer_by_degradation.png`,
  `results/figures/cer_vs_severity.png`.

## 5. Success cases

- Which degradations the pipeline fixes, with the size of the improvement.
- Why, in DIP terms (which stage does the work).
- A before/after figure.

## 6. Failure cases

- Which degradations the pipeline makes **worse**, with numbers. Do not hide these.
- The clean-page cost: what preprocessing does to an already-clean scan.
- The worst individual cases and what went wrong in each.
- *Assets:* `results/failures/`, `results/failures/failures.csv`.

## 7. Ablation

- Leave-one-stage-out on the headline pipeline.
- Which stages earn their place, which are neutral, which hurt.
- *Assets:* `results/ablation.csv`, `results/figures/ablation.png`.

## 8. Limitations

- Synthetic data; one OCR engine; one degradation at a time; fixed parameters;
  single page; Tesseract defaults.

## 9. Future work

- Document classifier, learned OCR, field extraction, RAG, agent loop, MCP tools.
- Point at `docs/EXTENDING.md` for where each one attaches.

## 10. Reproduction

- `make setup && make all`, runtime, and where the outputs land.
