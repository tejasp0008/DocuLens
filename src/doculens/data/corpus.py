"""Building and loading the degraded corpus and its manifests.

This module owns the only filesystem layout knowledge in the data package.
Degradation functions themselves stay pure and are applied here with a seed
derived from ``(global seed, doc id, degradation, level)``, so every degraded
file is reproducible independently of the order in which files are written.
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from doculens.config import Config
from doculens.data.degradations import apply_degradation

logger = logging.getLogger(__name__)

CLEAN_CONDITION = "none"
"""Degradation name used for the undegraded condition."""

CLEAN_LEVEL = 0
"""Severity level used for the undegraded condition."""

_FILENAME_SEPARATOR = "__"


class CorpusError(RuntimeError):
    """Raised when expected corpus files or manifests are missing."""


def degraded_filename(doc_id: str, degradation: str, level: int) -> str:
    """Return the filename for one degraded variant.

    Args:
        doc_id: Document identifier, for example ``doc_001``.
        degradation: Registered degradation name.
        level: Severity level.

    Returns:
        A name such as ``doc_001__blur__2.png``.
    """
    return f"{doc_id}{_FILENAME_SEPARATOR}{degradation}{_FILENAME_SEPARATOR}{level}.png"


def derive_variant_rng(
    global_seed: int, doc_id: str, degradation: str, level: int
) -> np.random.Generator:
    """Derive the stable random generator for one degraded variant.

    ``hash()`` is salted per process, so CRC32 is used instead to keep seeds
    identical across runs and machines.

    Args:
        global_seed: The configured global seed.
        doc_id: Document identifier.
        degradation: Degradation name.
        level: Severity level.

    Returns:
        A seeded generator unique to this variant.
    """
    entropy = [
        int(global_seed),
        int(zlib.crc32(doc_id.encode("utf-8"))),
        int(zlib.crc32(degradation.encode("utf-8"))),
        int(level),
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def load_clean_manifest(config: Config) -> pd.DataFrame:
    """Load the manifest written by corpus generation.

    Args:
        config: Validated configuration.

    Returns:
        The clean-corpus manifest.

    Raises:
        CorpusError: If the manifest does not exist.
    """
    path = config.paths.manifest_clean
    if not path.is_file():
        raise CorpusError(f"clean manifest not found at {path}; run `doculens generate` first")
    return pd.read_csv(path)


def load_degraded_manifest(config: Config) -> pd.DataFrame:
    """Load the manifest of degraded variants.

    Args:
        config: Validated configuration.

    Returns:
        The degraded-corpus manifest.

    Raises:
        CorpusError: If the manifest does not exist.
    """
    path = config.paths.manifest_degraded
    if not path.is_file():
        raise CorpusError(f"degraded manifest not found at {path}; run `doculens degrade` first")
    return pd.read_csv(path)


def read_image(path: Path) -> np.ndarray:
    """Read a grayscale image from disk.

    Args:
        path: Image path.

    Returns:
        A 2-D ``uint8`` array.

    Raises:
        CorpusError: If the file cannot be read as an image.
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise CorpusError(f"could not read image: {path}")
    return image


def build_degraded_corpus(config: Config, show_progress: bool = True) -> pd.DataFrame:
    """Apply every configured degradation at every severity to every document.

    The manifest also records an undegraded row per document (degradation
    ``none``, level ``0``) pointing at the clean PNG, so that evaluation can
    quantify what the pipeline does to a clean scan -- which is where
    preprocessing is most likely to hurt rather than help.

    Args:
        config: Validated configuration.
        show_progress: Whether to display a progress bar.

    Returns:
        The degraded-corpus manifest, one row per evaluated image.

    Raises:
        CorpusError: If the clean corpus has not been generated.
    """
    clean = load_clean_manifest(config)
    config.paths.ensure_dirs()

    tasks: list[tuple[str, str, str, str, int]] = []
    for row in clean.itertuples(index=False):
        tasks.append((row.doc_id, row.doc_type, row.split, CLEAN_CONDITION, CLEAN_LEVEL))
        for degradation in config.degradations:
            for level in sorted(config.degradations[degradation].levels):
                tasks.append((row.doc_id, row.doc_type, row.split, degradation, level))

    clean_png = dict(zip(clean["doc_id"], clean["png"]))
    clean_txt = dict(zip(clean["doc_id"], clean["txt"]))

    rows: list[dict[str, object]] = []
    iterator = tqdm(tasks, desc="degrading documents", unit="img") if show_progress else tasks
    for doc_id, doc_type, split, degradation, level in iterator:
        source = config.paths.root / clean_png[doc_id]
        if degradation == CLEAN_CONDITION:
            image_path = source
        else:
            image = read_image(source)
            rng = derive_variant_rng(config.seed, doc_id, degradation, level)
            params = config.degradations[degradation].params_for(level)
            degraded = apply_degradation(degradation, image, rng, params)
            image_path = config.paths.degraded_dir / degraded_filename(doc_id, degradation, level)
            cv2.imwrite(str(image_path), degraded)
        rows.append(
            {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "split": split,
                "degradation": degradation,
                "level": level,
                "image": str(image_path.relative_to(config.paths.root)),
                "txt": clean_txt[doc_id],
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(config.paths.manifest_degraded, index=False)
    logger.info(
        "wrote %d corpus images (%d documents x %d conditions) -> %s",
        len(manifest),
        clean.shape[0],
        len(manifest) // max(clean.shape[0], 1),
        config.paths.manifest_degraded,
    )
    return manifest


def select_evaluation_items(
    config: Config, splits: tuple[str, ...] | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Return the manifest rows to evaluate.

    Args:
        config: Validated configuration.
        splits: Splits to keep. Defaults to ``eval.report_splits``.
        limit: Optional cap on the number of rows, for smoke runs.

    Returns:
        The filtered manifest, sorted for stable iteration order.

    Raises:
        CorpusError: If the filter selects no rows.
    """
    manifest = load_degraded_manifest(config)
    wanted = tuple(splits) if splits is not None else config.eval.report_splits
    selected = manifest[manifest["split"].isin(wanted)].copy()
    if selected.empty:
        raise CorpusError(
            f"no corpus rows for splits {list(wanted)}; manifest has "
            f"{sorted(manifest['split'].unique())}"
        )
    selected = selected.sort_values(["doc_id", "degradation", "level"]).reset_index(drop=True)
    if limit is not None:
        selected = selected.head(int(limit)).reset_index(drop=True)
    return selected
