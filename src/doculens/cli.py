"""Single command-line entry point for the whole DocuLens pipeline.

Subcommands mirror the stages of the experiment, and ``all`` runs them in
order. Every subcommand takes the same ``--config``, so a variant experiment is
a copy of the YAML file rather than a code change.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from doculens.config import Config, ConfigError, load_config
from doculens.data.corpus import CorpusError, build_degraded_corpus, select_evaluation_items
from doculens.data.generator import generate_corpus
from doculens.eval.ablation import run_ablation
from doculens.eval.failures import generate_failure_assets
from doculens.eval.report import build_report
from doculens.eval.runner import default_jobs, run_and_save, summarise_overall
from doculens.logging_setup import configure_logging
from doculens.ocr.base import OCRError
from doculens.ocr.tesseract_engine import TesseractEngine, ensure_tesseract

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")

_OCR_COMMANDS = frozenset({"run", "ablate", "all"})

EXIT_OK = 0
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for every subcommand.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="doculens",
        description=(
            "Classical DIP preprocessing pipeline for OCR of Indian insurance claim "
            "documents: generate a synthetic corpus, degrade it, compare preprocessing "
            "pipelines against a raw baseline, and write the report assets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG, help="path to the YAML config"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument(
        "--no-progress", action="store_true", help="disable progress bars (useful in CI logs)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate the synthetic clean corpus")
    generate.add_argument(
        "--n-docs", type=int, default=None, help="override data.n_docs for a quick run"
    )

    subparsers.add_parser("degrade", help="apply every degradation at every severity level")

    run = subparsers.add_parser("run", help="evaluate every method over the corpus")
    _add_eval_arguments(run)
    run.add_argument(
        "--methods", nargs="+", default=None, help="override eval.methods for this run"
    )

    ablate = subparsers.add_parser("ablate", help="leave-one-stage-out ablation")
    _add_eval_arguments(ablate)
    ablate.add_argument(
        "--pipeline", default=None, help="override eval.ablation.pipeline for this run"
    )

    subparsers.add_parser("report", help="write tables, charts and figures from existing results")

    everything = subparsers.add_parser(
        "all", help="generate, degrade, run, ablate and report in one go"
    )
    _add_eval_arguments(everything)
    everything.add_argument("--n-docs", type=int, default=None, help="override data.n_docs")
    everything.add_argument(
        "--skip-ablation", action="store_true", help="skip the ablation stage (it is the slowest)"
    )
    return parser


def _add_eval_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by every OCR-running subcommand.

    Args:
        parser: Subcommand parser to extend.
    """
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help=f"OCR worker threads (default: {default_jobs()} on this machine)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="override eval.report_splits (reported numbers should use test only)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="evaluate at most this many corpus images"
    )


def _engine(config: Config) -> TesseractEngine:
    """Build the OCR engine named in the config.

    Args:
        config: Validated configuration.

    Returns:
        The configured engine.
    """
    return TesseractEngine.from_config(config.ocr.tesseract)


def _selected_manifest(config: Config, args: argparse.Namespace) -> pd.DataFrame:
    """Select the corpus rows an OCR subcommand should evaluate.

    Args:
        config: Validated configuration.
        args: Parsed arguments.

    Returns:
        The selected manifest rows.
    """
    splits = tuple(args.splits) if getattr(args, "splits", None) else None
    return select_evaluation_items(config, splits=splits, limit=getattr(args, "limit", None))


def command_generate(config: Config, args: argparse.Namespace) -> int:
    """Generate the synthetic clean corpus.

    Args:
        config: Validated configuration.
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    config.seed_everything()
    manifest = generate_corpus(config, n_docs=args.n_docs, show_progress=not args.no_progress)
    counts = manifest["split"].value_counts().sort_index()
    print(f"\nGenerated {len(manifest)} documents in {config.paths.clean_dir}")
    print("  split sizes: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  manifest:    {config.paths.manifest_clean}")
    return EXIT_OK


def command_degrade(config: Config, args: argparse.Namespace) -> int:
    """Apply every configured degradation to the clean corpus.

    Args:
        config: Validated configuration.
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    config.seed_everything()
    manifest = build_degraded_corpus(config, show_progress=not args.no_progress)
    print(f"\nBuilt {len(manifest)} corpus images in {config.paths.degraded_dir}")
    print(f"  conditions: {manifest.groupby(['degradation', 'level']).ngroups} per document")
    print(f"  manifest:   {config.paths.manifest_degraded}")
    return EXIT_OK


def command_run(config: Config, args: argparse.Namespace) -> int:
    """Evaluate every configured method over the corpus.

    Args:
        config: Validated configuration.
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    config.seed_everything()
    manifest = _selected_manifest(config, args)
    results = run_and_save(
        config=config,
        manifest=manifest,
        engine=_engine(config),
        methods=args.methods,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )
    predictions = pd.read_csv(config.paths.results_dir / "predictions.csv")
    failures = generate_failure_assets(config, results, predictions, manifest)
    _print_overall(results)
    print(f"\n  results:  {config.paths.results_long}")
    print(f"  failures: {len(failures)} figures in {config.paths.failures_dir}")
    return EXIT_OK


def command_ablate(config: Config, args: argparse.Namespace) -> int:
    """Run the leave-one-stage-out ablation.

    Args:
        config: Validated configuration.
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    config.seed_everything()
    manifest = _selected_manifest(config, args)
    summary, _ = run_ablation(
        config=config,
        manifest=manifest,
        engine=_engine(config),
        pipeline_name=args.pipeline,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )
    print("\nAblation (positive delta = removing the stage made CER worse):")
    for row in summary.itertuples(index=False):
        label = row.removed_stage or "(full pipeline)"
        print(f"  {label:<24s} CER {row.cer_mean:.4f}   delta {row.delta_cer_vs_full:+.4f}")
    print(f"\n  ablation: {config.paths.ablation_csv}")
    return EXIT_OK


def command_report(config: Config, args: argparse.Namespace) -> int:
    """Write every report asset from the results already on disk.

    Args:
        config: Validated configuration.
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    del args
    written = build_report(config)
    print(f"\nWrote {len(written)} report assets:")
    for path in written:
        print(f"  {path.relative_to(config.paths.root)}")
    return EXIT_OK


def command_all(config: Config, args: argparse.Namespace) -> int:
    """Run the whole experiment end to end.

    Args:
        config: Validated configuration.
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    config.seed_everything()
    generate_corpus(config, n_docs=args.n_docs, show_progress=not args.no_progress)
    build_degraded_corpus(config, show_progress=not args.no_progress)

    manifest = _selected_manifest(config, args)
    engine = _engine(config)
    results = run_and_save(
        config=config,
        manifest=manifest,
        engine=engine,
        jobs=args.jobs,
        show_progress=not args.no_progress,
    )
    if not args.skip_ablation:
        run_ablation(
            config=config,
            manifest=manifest,
            engine=engine,
            jobs=args.jobs,
            show_progress=not args.no_progress,
        )
    predictions = pd.read_csv(config.paths.results_dir / "predictions.csv")
    generate_failure_assets(config, results, predictions, manifest)
    build_report(config)

    _print_overall(results)
    _print_headline_by_degradation(config, results)
    print(f"\n  results directory: {config.paths.results_dir}")
    return EXIT_OK


def _print_overall(results: pd.DataFrame) -> None:
    """Print the per-method summary table to stdout.

    Args:
        results: Tidy long-format results.
    """
    overall = summarise_overall(results)
    print("\nMean CER / WER per method (lower is better):")
    print(f"  {'method':<22s} {'CER':>8s} {'WER':>8s} {'ms/img':>9s}")
    for row in overall.itertuples(index=False):
        print(
            f"  {row.method:<22s} {row.cer_mean:8.4f} {row.wer_mean:8.4f} "
            f"{row.runtime_ms_mean:9.1f}"
        )


def _print_headline_by_degradation(config: Config, results: pd.DataFrame) -> None:
    """Print baseline versus headline-pipeline CER for each degradation.

    Args:
        config: Validated configuration.
        results: Tidy long-format results.
    """
    from doculens.config import BASELINE_METHOD

    headline = config.eval.report.headline_method
    pivot = results.pivot_table(index="degradation", columns="method", values="cer", aggfunc="mean")
    if BASELINE_METHOD not in pivot.columns or headline not in pivot.columns:
        return
    print(f"\nMean CER by degradation ({BASELINE_METHOD} vs {headline}):")
    print(f"  {'degradation':<14s} {'baseline':>10s} {headline:>20s} {'change':>10s}")
    for degradation, row in pivot.iterrows():
        base, head = float(row[BASELINE_METHOD]), float(row[headline])
        change = f"{(head - base) / base * 100:+.1f}%" if base else "n/a"
        print(f"  {degradation:<14s} {base:10.4f} {head:20.4f} {change:>10s}")


_COMMANDS = {
    "generate": command_generate,
    "degrade": command_degrade,
    "run": command_run,
    "ablate": command_ablate,
    "report": command_report,
    "all": command_all,
}


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 2 on a configuration, corpus or OCR
        error (all of which are reported as a plain message, not a traceback).
    """
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    try:
        config = load_config(args.config)
        if args.command in _OCR_COMMANDS:
            ensure_tesseract()
        return _COMMANDS[args.command](config, args)
    except (ConfigError, CorpusError, OCRError, FileNotFoundError, ValueError) as exc:
        logger.debug("command failed", exc_info=True)
        print(f"\nerror: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("\ninterrupted", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
