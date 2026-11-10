# DocuLens

**Classical digital image processing for OCR of Indian insurance claim documents.**

Claim intake receives claim forms and garage invoices as phone photos and rescans:
out of focus, noisy, skewed, unevenly lit, heavily recompressed. DocuLens asks a
measurable question:

> How much OCR accuracy can classical DIP recover before the recognition engine
> ever sees the page — and where does preprocessing make things *worse*?

To answer it the project generates a synthetic corpus with **exact** ground truth,
degrades it under control, runs a raw baseline against swappable preprocessing
pipelines, and reports CER and WER with an ablation and a failure analysis.

---

---

## Headline results

Test split only (18 documents × 16 capture conditions = 288 images, 5 methods,
1440 OCR calls). Full tables in `results/tables.md`, generated prose in
`results/REPORT.md`.

| method | mean CER | mean WER |
|---|---|---|
| `baseline_raw` (no preprocessing) | 0.0749 | 0.1082 |
| **`pipeline_full`** | **0.0089** | **0.0898** |
| `pipeline_clahe` | 0.0090 | 0.0765 |
| `pipeline_basic` | 0.0796 | 0.1439 |
| `pipeline_full_otsu` | 0.0984 | 0.2567 |

**Mean CER drops 88%, from 0.0749 to 0.0089.** But the average hides the shape of
the result, which is the interesting part:

| degradation | baseline | `pipeline_full` | |
|---|---|---|---|
| lowlight | 0.3688 | **0.0087** | −97.6% |
| skew | 0.0183 | **0.0059** | −67.9% |
| noise | 0.0119 | **0.0080** | −33.1% |
| blur | **0.0002** | 0.0169 | *worse* |
| jpeg | **0.0003** | 0.0052 | *worse* |
| none (clean page) | **0.0003** | 0.0084 | *worse* |

Read honestly, that says:

- **Preprocessing rescues genuinely broken captures.** Under a shadow gradient the
  raw baseline loses more than a third of all characters (CER 0.37, rising to 0.58
  at severity 3); adaptive thresholding brings it back to 0.009 — effectively a
  clean read. Deskew does the same for rotated pages.
- **Preprocessing costs accuracy on pages that were already fine.** Tesseract reads
  a clean, blurred or recompressed synthetic page near-perfectly (CER ≈ 0.0002).
  Binarising it can only lose information, and does: CER rises to 0.008–0.017. The
  percentages look enormous because the baseline is near zero; in absolute terms it
  is roughly one extra error per 120 characters.
- **Global thresholding is the wrong tool here.** `pipeline_full_otsu` is the worst
  method overall (0.0984) because a single global threshold cannot survive the
  shadow gradient — it scores 0.4235 on `lowlight`, worse than no preprocessing.

The practical reading: **a document-quality classifier should gate the pipeline**,
applying it only to pages that need it. That is the first thing `docs/EXTENDING.md`
describes.

### What the ablation says

Leave-one-stage-out on `pipeline_full` (`results/figures/ablation.png`). Positive
delta = removing the stage made CER worse, i.e. the stage earns its place.

| removed stage | mean CER | Δ vs full |
|---|---|---|
| `adaptive_threshold` | 0.0858 | **+0.0769** |
| `nlmeans_denoise` | 0.0213 | **+0.0124** |
| `deskew` | 0.0136 | **+0.0047** |
| `grayscale` | 0.0089 | +0.0000 |
| `bilateral_filter` | 0.0078 | −0.0011 |
| `morphology_open` | 0.0050 | −0.0039 |
| `median_blur` | 0.0028 | −0.0061 |

Adaptive thresholding does almost all of the work. `grayscale` is a no-op because
the corpus is already single-channel (it is kept because colour input is the
realistic case). And **three stages look net-negative**: removing `median_blur`,
`bilateral_filter` or `morphology_open` each *lowers* test-split CER.

### Why that ablation must not be acted on

The obvious move is to delete those three stages. Doing it would have been a
mistake, for two separate reasons — and this is the most useful thing the project
measured.

**First, it would be test-set leakage.** Picking a configuration because it won on
the split the headline numbers come from invalidates those numbers. So the lean
variant was defined as `pipeline_lean` in the config, kept out of `eval.methods`,
and checked on the **train** split instead:

```bash
python scripts/tune_pipeline.py --pipelines pipeline_full pipeline_lean --docs 10
```

| method | train-split mean CER |
|---|---|
| `pipeline_full` | **0.0085** |
| `pipeline_lean` (the three stages removed) | 0.0460 |
| `baseline_raw` | 0.0745 |

`pipeline_lean` is **five times worse**, and the damage is concentrated in one
cell: σ=40 sensor noise, where it scores **0.7054** against `pipeline_full`'s
0.0068. The ablation's suggestion was wrong.

**Second, leave-one-out measures marginal contribution, not joint necessity.** The
denoising stages are partly redundant with each other: with non-local means,
median and bilateral filtering all present, removing any *single* one changes
little, so each looks droppable. Remove all three and heavy noise survives into the
binarisation step and destroys the page. A leave-one-out table can never show that,
because it never tests the combination.

The stages stay. The honest summary is that `median_blur`, `bilateral_filter` and
`morphology_open` cost a little accuracy on easy pages and are load-bearing on hard
ones.

---

## Quick start

```bash
make setup     # virtualenv + pinned dependencies + Tesseract check
make all       # generate, degrade, evaluate, ablate, analyse failures, report
```

`make all` is fully seeded and reproduces the same numbers on a clean checkout.
On a 10-core laptop it takes about 28 minutes, almost all of it Tesseract
(1440 + 2304 OCR calls); `--skip-ablation` cuts that to about 10.

Outputs land in `results/` — see [Where the outputs go](#where-the-outputs-go).

### Prerequisites

- **Python 3.10+**
- **Tesseract OCR** on `PATH`. DocuLens checks for it at startup and fails with
  install instructions rather than part-way through a run.

| Platform | Install |
|---|---|
| macOS (Homebrew) | `brew install tesseract` |
| Debian / Ubuntu | `sudo apt-get update && sudo apt-get install -y tesseract-ocr` |
| Fedora / RHEL | `sudo dnf install -y tesseract` |
| Arch | `sudo pacman -S tesseract tesseract-data-eng` |
| Windows | `winget install -e --id UB-Mannheim.TesseractOCR` |

Verify with `tesseract --version`.

### Manual setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
doculens --help
```

---

## Commands

A single entry point, `doculens`, with one subcommand per stage of the experiment.
Global options (`-c/--config`, `-v/--verbose`, `--no-progress`) come **before** the
subcommand.

| Command | What it does | Make target |
|---|---|---|
| `doculens generate` | Render the synthetic clean corpus + ground truth | `make generate` |
| `doculens degrade` | Apply every degradation at every severity level | `make degrade` |
| `doculens run` | Evaluate baseline and pipelines on the test split | `make run` |
| `doculens ablate` | Leave-one-stage-out ablation | `make ablate` |
| `doculens report` | Tables, charts, figures, filled report | `make report` |
| `doculens all` | All of the above in order | `make all` |

Useful flags:

```bash
doculens generate --n-docs 5               # small corpus for a quick look
doculens run --jobs 4                      # OCR worker threads
doculens run --limit 40                    # cap images, for a smoke test
doculens run --splits train val            # evaluate other splits (not for reporting)
doculens ablate --pipeline pipeline_basic  # ablate a different pipeline
doculens all --skip-ablation               # the ablation is the slow part
doculens -c configs/my_experiment.yaml all # a variant experiment is a copy of the YAML
```

Also:

```bash
make test    # pytest
make lint    # ruff check + ruff format --check
make clean   # delete the generated dataset and results
```

---

---

## Live demo UI

For presenting the pipeline interactively rather than through static figures:

```bash
make demo          # http://127.0.0.1:8000  (installs flask on first run)
make demo PORT=9000
```

The page has two tabs.

### Upload images (default tab)

Drop in one or more document images, choose a pipeline and press **Run full
pipeline**. Each file is processed in about 1.3 s and its row appears as it
finishes, so a batch fills in live. The result is:

- **A results table** — one row per file with baseline CER, pipeline CER, the
  signed delta, baseline and pipeline OCR confidence, words read, preprocessing
  time and a green/red verdict, plus a mean row across the batch. **Download
  results CSV** saves exactly those numbers.
- **A per-file breakdown** — click any row (or the file's heading) to open its
  scoreboard, its full stage filmstrip and its three transcripts.

Two kinds of metric appear there, and the distinction matters:

| | needs ground truth | what it measures |
| --- | --- | --- |
| **CER / WER** | yes | edit distance to the exact reference text |
| **OCR confidence** | no | how sure Tesseract is, *not* how right it is |

A file whose name follows the corpus convention — `doc_002.png` or
`doc_002__lowlight__3.png` — is recognised as a page this project generated, so
its reference text is loaded from disk and CER is exact. Any other image is
marked `no ground truth`, its CER columns show `—`, and only confidence, word
count and the stage images describe it. Confidence is never used to choose a
pipeline or quoted as a stand-in for CER: a confidently misread word scores
high.

For a live demonstration with real numbers, upload straight out of
`data/degraded/` (for example `doc_002__lowlight__3.png`,
`doc_008__skew__2.png`, `doc_018__blur__3.png`): the names match, so every row is
scored, and `doc_018__blur__3.png` is one of the six worst cases in
`results/REPORT.md` — a useful thing to be the one to point out.

### Controlled corpus

Pick a generated document, a degradation and a severity, choose a pipeline and
hit **Run**. This is the measured path — the degradation is applied with the same
seeded RNG the corpus used, so the CER shown for, say, `doc_002` / lowlight / 3
is the same number that image contributes to `results/results_long.csv`. It shows:

- **The scoreboard** — raw-baseline CER/WER beside the pipeline's, with the verdict
  coloured green or red. This is the whole project in one glance, and it will
  honestly go red on a clean page.
- **The stage filmstrip** — every intermediate image, labelled with the stage name
  and its parameters. Click any frame to enlarge it full-screen (Esc to close);
  this is the best way to show the shadow gradient surviving to `deskew` and
  disappearing at `adaptive_threshold`.
- **Three transcripts** — ground truth, baseline OCR and pipeline OCR, normalised
  exactly as the metrics see them, so the specific character confusions are
  readable.

The demo is a **presentation harness, not part of the graded pipeline**: it lives
in `demo/`, adds no image-processing logic, and calls the same registry, `Pipeline`,
`OCREngine` and `metrics.score` the evaluation uses. That it can be this thin is
the architectural point — those functions take plain images and return plain
values, which is also what the future MCP tool layer needs.

Flask is the only extra dependency and is deliberately kept out of
`requirements.txt`, so `make all` never needs it.

## Dataset

Real claim documents cannot be published with a course report, and scanned
documents have no exact ground truth. DocuLens therefore renders its own corpus,
which is what makes CER and WER meaningful rather than approximate.

**60 documents**, two layouts, alternating:

- **`claim_form`** — motor insurance claim form: policy and claim numbers, insured
  person, address, vehicle details, loss details, declaration.
- **`invoice`** — garage tax invoice: invoice number, GSTIN, customer, vehicle,
  labour/parts charges, CGST/SGST, total, terms.

Content comes from Faker's `en_IN` provider plus project-specific generators for
Indian formats: vehicle registrations (`MH 12 AB 3456`), policy numbers
(`POL-MH-2023-004821`), claim numbers (`CLM/2024/0481234`), GSTINs, and rupee
amounts with Indian digit grouping (`INR 1,23,456.50`). Invoice arithmetic is
internally consistent: subtotal = labour + parts, CGST = SGST = 9%, total = sum.

Each document is written three ways:

| File | Contents |
|---|---|
| `data/clean/doc_NNN.png` | 1240×1754 rendered page |
| `data/clean/doc_NNN.txt` | **exact** rendered text — the ground truth |
| `data/clean/doc_NNN.json` | structured `{field: value}` — supervision for future extraction work |

### Splits

Recorded in `data/manifest_clean.csv` **before any tuning**: 50% train (30 docs),
20% val (12), 30% test (18). **Every reported number uses the test split only.**
Pipeline parameters were chosen on the train split with
[`scripts/tune_pipeline.py`](scripts/tune_pipeline.py), which is kept in the
repository so the tuning is auditable.

### Degradations

Five degradations × three severity levels, plus the undegraded `none` condition —
**16 conditions per document, 960 corpus images**. Files are named
`doc_001__blur__2.png` and indexed in `data/manifest_degraded.csv`.

| Degradation | Simulates | Level 1 | Level 2 | Level 3 |
|---|---|---|---|---|
| `blur` | defocused phone photo | Gaussian k=3 | k=5 | k=7 |
| `noise` | low-light sensor noise | σ=10 | σ=25 | σ=40 |
| `skew` | page fed in crooked | 2° | 5° | 8° |
| `lowlight` | desk-lamp shadow gradient | gain 0.75→1.0 | 0.55→1.0 | 0.40→1.0 |
| `jpeg` | forwarded over chat apps | quality 60 | 30 | 10 |

Each is seeded from `(global seed, doc id, degradation, level)`, so a file is
reproducible regardless of the order anything runs in.

---

## Method

### Pipelines

A pipeline is an **ordered list of stages in YAML** — never hard-coded. Compared
methods:

| Method | Stages |
|---|---|
| `baseline_raw` | *(none — Tesseract on the raw image)* |
| `pipeline_basic` | grayscale → median blur → Otsu threshold |
| `pipeline_full` | grayscale → non-local means → median blur → bilateral filter → deskew → adaptive threshold → morphological opening |
| `pipeline_full_otsu` | as `pipeline_full` but global Otsu instead of adaptive |
| `pipeline_clahe` | as `pipeline_full` but with CLAHE added after denoising |

Available stages: `grayscale`, `clahe`, `median_blur`, `bilateral_filter`,
`nlmeans_denoise`, `otsu_threshold`, `adaptive_threshold`, `deskew`,
`morphology_open`, `perspective_correct`.

Two stages are there because the train-split sweep
([`scripts/tune_pipeline.py`](scripts/tune_pipeline.py)) said so, and one famous
stage is absent for the same reason:

- **`nlmeans_denoise` is in**, because median and bilateral filtering alone could not
  cope with σ=40 sensor noise — the pipeline was *worse* than the raw baseline there
  until non-local means was added.
- **`clahe` is out.** CLAHE was in the first draft of the pipeline to handle the
  shadow gradient. On the train split it turned out to *amplify sensor noise* badly
  enough to break the binarisation that follows it — mean CER on σ=25 noise went
  from 0.008 to 0.77 — while adaptive thresholding already removed the gradient on
  its own. Once non-local means denoising was added in front of it, CLAHE stopped
  being harmful but still added nothing, so it is out of `pipeline_full` and kept as
  the separate `pipeline_clahe` method, measured rather than quietly dropped.
- **`median_blur` uses ksize 3, not 5.** A 5-pixel median erases thin glyph strokes
  at this resolution; the sweep showed it roughly five times worse than ksize 3.

Two notes on stages that are easy to get wrong:

- **`deskew`** estimates the angle from `cv2.minAreaRect` over the inverted
  foreground. OpenCV changed that function's angle range at 4.5, so the raw angle
  is folded into `[-45°, 45°]`, which is correct under either convention; an
  estimate larger than `max_abs_angle_deg` is discarded rather than applied. A unit
  test rotates a synthetic page by ±2/5/8° and asserts it comes back level within
  0.5°.
- **`morphology_open`** operates on the *inverted* image by default, because these
  pages are dark text on light paper. Opening the un-inverted image would act on
  the paper, not the glyphs.

### Metrics

CER and WER are edit-distance ratios computed with `jiwer`, after **the same
normalisation is applied to both the ground truth and the OCR output**:

| Step | Default | Effect |
|---|---|---|
| lowercase | on | case differences are not counted as errors |
| collapse whitespace | on | every run of spaces/newlines becomes one space |
| strip | on | leading/trailing whitespace ignored |
| remove punctuation | **off** | punctuation errors *are* counted |

Collapsing whitespace is what lets the two-column rendering `Label:        value`
be scored against the ground-truth line `Label: value`. All four steps are config
switches under `eval.normalisation`.

Conventions worth stating because they affect the numbers:

- Identical strings score **0.0**; an empty reference against a non-empty
  hypothesis scores **1.0**.
- **CER is not capped at 1.0.** An engine that hallucinates text across a noisy
  blank region produces more characters than the reference, and that is reported
  as-is rather than clipped — clipping would hide exactly the failure mode worth
  reporting.

OCR runs as `tesseract --oem 3 --psm 6` (`--psm 6` = "assume a single uniform block
of text"), configured under `ocr.tesseract`.

---

## Where the outputs go

```
data/
  clean/          doc_NNN.png | .txt (ground truth) | .json (fields)
  degraded/       doc_NNN__<degradation>__<level>.png
  manifest_clean.csv, manifest_degraded.csv
outputs/
  stages/<doc>/   00_input.png, 01_grayscale.png, ... (per-stage images)
results/
  results_long.csv            tidy: doc_id, degradation, level, method, cer, wer, runtime_ms
  predictions.csv             every OCR transcript
  summary_by_degradation.csv, summary_by_level.csv, summary_overall.csv
  ablation.csv, ablation_long.csv
  tables.md                   every table as markdown, ready to paste
  REPORT.md                   the report skeleton filled with real numbers
  figures/                    charts, stage montages, block diagram (200 dpi)
  failures/                   worst-6 side-by-side figures + failures.csv
```

`data/`, `outputs/` and `results/` are gitignored; `make all` regenerates them.

---

## Repository layout

```
src/doculens/
  cli.py            single entry point (argparse subcommands)
  config.py         dataclasses + YAML loading + validation
  data/
    generator.py    synthetic documents + ground truth
    degradations.py degradation registry
    corpus.py       degraded-corpus construction and manifests
  dip/
    base.py         Stage protocol + BaseStage
    stages.py       the DIP stages
    registry.py     name -> Stage class; builds pipelines from config
    pipeline.py     Pipeline, stage capture, leave-one-out
    segmentation.py connected-component field regions
  ocr/
    base.py         OCREngine protocol
    tesseract_engine.py
  eval/
    metrics.py      normalisation, CER, WER
    runner.py       runs methods over the corpus -> tidy rows
    ablation.py     leave-one-stage-out
    failures.py     worst-k selection + figures
    charts.py       chart primitives and theme
    report.py       tables, charts, figures, filled report
  extraction/ rag/ services/    STUBS ONLY - see docs/EXTENDING.md
configs/default.yaml            every tunable value in the project
scripts/tune_pipeline.py        train-split parameter sweep
demo/                           live demo UI (presentation only, not graded)
tests/                          pytest suite
```

### Design rules

- **All parameters live in YAML.** No tuning constants in code; stage constructors
  take explicit keyword arguments, so an unknown key in the config raises at load
  time instead of being silently ignored.
- **Everything interchangeable is behind a registry or Protocol** — stages,
  degradations, OCR engines, and the extraction/RAG/service stubs.
- **Pipelines and OCR are pure.** No mutation of inputs, no filesystem access; the
  single exception is `Pipeline.run_saving_stages`.
- **Results are tidy long-format rows**, so every table and chart is a `groupby` of
  one CSV and a new method needs no reporting-code changes.
- **Deterministic**: one seed in the config drives Faker, numpy and `random`.

---

## Extending

`docs/EXTENDING.md` describes how each planned component attaches without
rewriting the core: a new DIP stage, a learned OCR engine (PaddleOCR/TrOCR) behind
`OCREngine`, a CNN/ViT document classifier, field extraction, RAG over policy
rules, and an MCP server whose six tools (`enhance_image`, `run_ocr`,
`classify_document`, `extract_fields`, `search_policy_rules`, `validate_claim`)
are thin wrappers over functions that already exist here. **None of these is
implemented** — `extraction/`, `rag/` and `services/` contain interface stubs that
raise `NotImplementedError`.

---

## Limitations

These bound what the numbers in this repository mean.

- **The data is synthetic.** Pages are rendered, not scanned. Real captures add
  paper texture, uneven ink, handwriting, stamps, staples, folds, shadows with
  structure and genuine perspective distortion. Absolute CER here is optimistic;
  the *relative* comparison between methods is the useful part.
- **One OCR engine.** Every number is Tesseract 5 with `--oem 3 --psm 6`. Tesseract
  is a classical engine that wants clean binarised input, so these pipelines flatter
  it. A transformer OCR model trained on degraded crops would likely prefer the raw
  image, and the ranking here would change.
- **One degradation at a time.** Real photographs are blurred *and* noisy *and*
  skewed at once; interactions are not measured.
- **Fixed parameters, lightly tuned.** Stage parameters came from a small sweep on
  the train split, not an exhaustive search. A per-degradation parameter sweep
  would improve every pipeline number.
- **Single page per document**, English only, no tables or logos in the layouts.
- **Timing is indicative.** `runtime_ms` is wall-clock per image under the
  configured thread count, so it compares methods fairly within a run but is not a
  single-core benchmark.

## Licence

MIT.
