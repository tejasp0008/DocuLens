"""Compare named pipelines on the TRAIN split, without touching the test results.

This script is how the stage choices in ``configs/default.yaml`` were made, and it
is kept in the repository so that the tuning is auditable rather than asserted.

Two properties matter:

* It filters the manifest to ``split == "train"``, so nothing it reports can leak
  the test split that the headline numbers come from.
* It writes nothing to ``results/``. It prints a table and exits, so running it
  can never overwrite a completed evaluation.

Usage:
    python scripts/tune_pipeline.py --pipelines pipeline_full pipeline_lean
    python scripts/tune_pipeline.py --docs 10 --jobs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doculens.config import BASELINE_METHOD, Config, load_config  # noqa: E402
from doculens.dip.pipeline import Pipeline  # noqa: E402
from doculens.eval.runner import EvalItem, run_evaluation  # noqa: E402
from doculens.logging_setup import configure_logging  # noqa: E402
from doculens.ocr.tesseract_engine import TesseractEngine, ensure_tesseract  # noqa: E402

TRAIN_SPLIT = "train"
DEFAULT_PIPELINES = ("pipeline_full", "pipeline_lean")


def train_items(config: Config, n_docs: int) -> list[EvalItem]:
    """Return evaluation items for the first ``n_docs`` train-split documents.

    Args:
        config: Validated configuration.
        n_docs: How many train documents to include.

    Returns:
        The evaluation items.

    Raises:
        SystemExit: If the corpus has not been built yet.
    """
    manifest_path = config.paths.manifest_degraded
    if not manifest_path.is_file():
        raise SystemExit(
            f"{manifest_path} not found; run `doculens generate` then `doculens degrade`"
        )
    manifest = pd.read_csv(manifest_path)
    train = manifest[manifest["split"] == TRAIN_SPLIT]
    if train.empty:
        raise SystemExit("no train-split rows in the manifest")
    docs = sorted(train["doc_id"].unique())[:n_docs]
    subset = train[train["doc_id"].isin(docs)]
    truth = {
        row.doc_id: (config.paths.root / row.txt).read_text(encoding="utf-8")
        for row in subset.itertuples(index=False)
    }
    return [
        EvalItem(
            doc_id=row.doc_id,
            doc_type=row.doc_type,
            split=row.split,
            degradation=row.degradation,
            level=int(row.level),
            image_path=config.paths.root / row.image,
            ground_truth=truth[row.doc_id],
        )
        for row in subset.itertuples(index=False)
    ]


def main() -> int:
    """Run the comparison and print a per-condition CER table.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=list(DEFAULT_PIPELINES),
        help="names of pipelines in the config to compare against the raw baseline",
    )
    parser.add_argument("--docs", type=int, default=10, help="train documents to compare over")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    configure_logging()
    ensure_tesseract()
    config = load_config(args.config)
    engine = TesseractEngine.from_config(config.ocr.tesseract)

    methods: dict[str, Pipeline | None] = {BASELINE_METHOD: None}
    for name in args.pipelines:
        methods[name] = config.build_pipeline(name)

    items = train_items(config, args.docs)
    results, _ = run_evaluation(
        items=items,
        methods=methods,
        engine=engine,
        normalisation=config.eval.normalisation,
        jobs=args.jobs,
        description="train-split comparison",
    )
    results["condition"] = results["degradation"] + "_" + results["level"].astype(str)
    table = results.pivot_table(
        index="condition", columns="method", values="cer", aggfunc="mean"
    ).round(4)[list(methods)]

    documents = results["doc_id"].nunique()
    print(f"\nTRAIN-split mean CER over {documents} documents (test split untouched)\n")
    print(table.to_string())
    print("\nOVERALL mean CER")
    print(table.mean().round(4).sort_values().to_string())
    print("\nNothing was written to results/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
