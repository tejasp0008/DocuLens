# Extending DocuLens

This project deliberately stops at *classical preprocessing + OCR + measurement*.
The longer-term system it is a foundation for — a document-understanding agent with
RAG over policy rules, exposed as MCP tools — is **not implemented here**. What *is*
here is the set of seams it plugs into.

This document says, for each planned piece, exactly where it attaches and what it
must not require you to change.

---

## The design rules that make this possible

1. **Pipelines live in YAML, not in code.** A pipeline is an ordered list of
   `{stage, params}` entries in `configs/default.yaml`. Nothing imports a pipeline
   by name.
2. **Everything interchangeable is behind a registry or a Protocol.** Stages
   (`STAGE_REGISTRY`), degradations (`DEGRADATION_REGISTRY`), OCR engines
   (`OCREngine`), extractors (`FieldExtractor`), retrievers (`Retriever`).
3. **Pure functions take images and strings, not paths.** `Pipeline.__call__`,
   `OCREngine.read`, `evaluate_image` and `metrics.score` make no filesystem
   assumptions, so a tool call or an agent step can invoke them directly. The only
   side-effecting method in the DIP layer is `Pipeline.run_saving_stages`.
4. **Results are tidy long-format rows.** Every table and chart is a `groupby` of
   `results/results_long.csv`, so a new method is a new value in the `method`
   column — not a new code path in the reporting layer.

---

## 1. Adding a DIP stage

**Touches:** `src/doculens/dip/stages.py`, `src/doculens/dip/registry.py`, the config.

```python
class UnsharpMaskStage(BaseStage):
    name: ClassVar[str] = "unsharp_mask"

    def __init__(self, radius: int, amount: float) -> None:
        require_odd_positive(self.name, "radius", radius)
        super().__init__(radius=int(radius), amount=float(amount))

    def apply(self, image: Image) -> Image:
        blurred = cv2.GaussianBlur(image, (self._params["radius"],) * 2, 0)
        return cv2.addWeighted(
            image, 1 + self._params["amount"], blurred, -self._params["amount"], 0
        )
```

Register it (`STAGE_REGISTRY[UnsharpMaskStage.name] = UnsharpMaskStage`, or call
`register_stage`), then use it in a pipeline:

```yaml
pipelines:
  pipeline_sharpen:
    - stage: grayscale
    - stage: unsharp_mask
      params: {radius: 5, amount: 0.8}
```

Add `pipeline_sharpen` to `eval.methods` and it appears in every table, chart and
the ablation automatically. **No evaluation or reporting code changes.**

Note that stage constructors take explicit keyword arguments on purpose: an
unknown key in the YAML raises at config-load time rather than being silently
dropped.

---

## 2. Adding an OCR engine (PaddleOCR, TrOCR)

**Touches:** a new file in `src/doculens/ocr/`, plus `ocr.engine` in the config.

The evaluation runner never imports Tesseract — it is handed an `OCREngine`:

```python
# src/doculens/ocr/trocr_engine.py
@dataclass(frozen=True)
class TrOCREngine:
    name: str = "trocr"
    model_id: str = "microsoft/trocr-base-printed"

    def read(
        self, image: np.ndarray
    ) -> str: ...  # must return "" rather than raise when nothing is legible
```

Then widen the engine check in `config._validate` (it currently rejects anything
but `tesseract` so that a typo fails loudly) and select it in `_engine()` in
`cli.py`. `runner.py`, `metrics.py`, `ablation.py`, `failures.py` and `report.py`
are untouched.

**Expect the conclusions to change.** The pipelines here are tuned for Tesseract,
which is a classical engine that wants clean binarised input. A transformer OCR
model trained on degraded crops often does *better* on the raw image than on a
binarised one — which is exactly why the comparison is worth running.

---

## 3. Document classification (CNN / ViT)

**Attaches to:** the `doc_type` the generator already writes to the manifest and to
each `<doc_id>.json`.

The corpus is already labelled: `data/manifest_clean.csv` has a `doc_type` column
and a train/val/test split that was fixed before any tuning. A classifier can be
trained on `train`, selected on `val`, and reported on `test` without regenerating
anything.

The payoff is a **per-type pipeline**: an invoice (dense numeric columns) and a
claim form (sparse label/value lines) do not want identical preprocessing. The
hook is one function that maps a predicted type to a pipeline name already in the
config — no new machinery.

---

## 4. Field extraction

**Interface:** `doculens.extraction.base.FieldExtractor` (stub).

Two inputs already exist:

- **Supervision.** `generate_corpus` writes `<doc_id>.json` with the exact
  `{key: value}` fields for every document. Nothing in the current evaluation reads
  it; it exists for this step.
- **Regions.** `dip.segmentation.segment_field_regions` returns label and value
  boxes (verified to separate the two columns cleanly — see
  `results/figures/segmentation_regions.png`).

A first implementation can be regular expressions over the transcript; a second can
pair each label region with the value region to its right and OCR only that crop,
which is usually far more accurate than parsing a whole-page transcript.

Metric to add: field-level exact-match accuracy, as a sibling of `metrics.score`.

---

## 5. RAG over policy rules

**Interface:** `doculens.rag.base.Retriever` (stub).

Nothing in the current codebase depends on this, by design. A retriever needs a
corpus of policy wordings, a chunker, an index (vector, BM25 or hybrid) and a
`search(query, top_k) -> list[RetrievedChunk]`. Keep `RetrievedChunk.source`
populated: claim decisions must be able to cite the clause they relied on.

---

## 6. Agent loop and MCP server

**Interface:** `doculens.services.base.ToolService` (stub), with the planned tool
surface in `PLANNED_TOOLS`.

The MCP server **reimplements nothing**. Each tool is a thin wrapper:

| Planned MCP tool      | Wraps                                              | Status |
|-----------------------|----------------------------------------------------|--------|
| `enhance_image`       | `Pipeline.__call__` via `config.build_pipeline`     | ready  |
| `run_ocr`             | `OCREngine.read`                                    | ready  |
| `classify_document`   | future CNN/ViT classifier                           | §3     |
| `extract_fields`      | `FieldExtractor.extract`                            | §4     |
| `search_policy_rules` | `Retriever.search`                                  | §5     |
| `validate_claim`      | agent loop over the four tools above                | future |

Two of the six can be implemented today against existing, tested code. That is the
whole point of keeping `Pipeline` and `OCREngine` pure and path-free: a tool call
arrives with bytes, not a directory layout.

A sketch of the agent loop for `validate_claim`:

```
enhance_image -> run_ocr -> classify_document -> extract_fields
      -> search_policy_rules(query built from the extracted fields)
      -> LLM judges the claim against the retrieved clauses, citing them
```

---

## 7. What would need rewriting (be honest about this)

- **Multi-page documents.** Everything here assumes one page per document — the
  manifest, the ground truth and the metrics are all per-page. Multi-page would
  need a document/page distinction throughout.
- **Non-rectangular capture.** `perspective_correct` exists as a stage but the
  synthetic corpus has no projective distortion, so it is untested on real data.
- **Word-level confidences and geometry.** `OCREngine.read` returns a plain string.
  Field extraction will eventually want `image_to_data` output (per-word boxes and
  confidences); that means widening the protocol with a second method, which is a
  real interface change rather than a drop-in.
