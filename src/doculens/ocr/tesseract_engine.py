"""Tesseract-backed :class:`~doculens.ocr.base.OCREngine` implementation."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from statistics import fmean

import cv2
import numpy as np
import pytesseract

from doculens.config import TesseractConfig
from doculens.ocr.base import Image, OCRError

logger = logging.getLogger(__name__)

_VERSION_TIMEOUT_S = 15

INSTALL_HINTS = """
Tesseract OCR was not found on PATH. DocuLens cannot run OCR without it.

Install it, then re-run:
  macOS (Homebrew):      brew install tesseract
  Debian / Ubuntu:       sudo apt-get update && sudo apt-get install -y tesseract-ocr
  Fedora / RHEL:         sudo dnf install -y tesseract
  Arch Linux:            sudo pacman -S tesseract tesseract-data-eng
  Windows (winget):      winget install -e --id UB-Mannheim.TesseractOCR
  Windows (manual):      https://github.com/UB-Mannheim/tesseract/wiki
                         then add the install directory to PATH, or set
                         pytesseract.pytesseract.tesseract_cmd in your code.

Verify with:  tesseract --version
""".strip()


class TesseractMissingError(OCRError):
    """Raised when the ``tesseract`` binary is absent or not executable."""


def ensure_tesseract() -> str:
    """Check that the ``tesseract`` binary is present and runnable.

    Called at CLI startup for every subcommand that performs OCR, so that a
    missing dependency fails immediately with install instructions instead of
    after minutes of image generation.

    Returns:
        The first line of ``tesseract --version`` output.

    Raises:
        TesseractMissingError: If the binary is missing or does not run.
    """
    binary = shutil.which("tesseract")
    if binary is None:
        raise TesseractMissingError(INSTALL_HINTS)
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_TIMEOUT_S,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TesseractMissingError(f"{INSTALL_HINTS}\n\nUnderlying error: {exc}") from exc
    version = (completed.stdout or completed.stderr).strip().splitlines()[0]
    logger.info("found %s at %s", version, binary)
    return version


@dataclass(frozen=True)
class TesseractEngine:
    """Reads text with Tesseract via :mod:`pytesseract`.

    Attributes:
        lang: Tesseract language code, for example ``eng``.
        config: Raw Tesseract CLI flags, for example ``--oem 3 --psm 6``.
        timeout_s: Per-image timeout in seconds.
        name: Engine name as recorded in results.
    """

    lang: str
    config: str
    timeout_s: int
    name: str = "tesseract"

    @classmethod
    def from_config(cls, config: TesseractConfig) -> TesseractEngine:
        """Build an engine from the ``ocr.tesseract`` config section.

        Args:
            config: The parsed Tesseract settings.

        Returns:
            The configured engine.
        """
        return cls(lang=config.lang, config=config.config, timeout_s=config.timeout_s)

    def read(self, image: Image) -> str:
        """Recognise text in ``image``.

        Args:
            image: Grayscale or BGR ``uint8`` image. Not mutated.

        Returns:
            The recognised text, or ``""`` if Tesseract found nothing or timed
            out. A timeout is logged as a warning and treated as an empty read
            so that one pathological image cannot abort a whole evaluation.
        """
        if not isinstance(image, np.ndarray):
            raise OCRError(f"TesseractEngine.read expected a numpy array, got {type(image)!r}")
        prepared = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        try:
            text = pytesseract.image_to_string(
                prepared, lang=self.lang, config=self.config, timeout=self.timeout_s
            )
        except RuntimeError as exc:  # pytesseract raises RuntimeError on timeout
            logger.warning("tesseract timed out after %ss: %s", self.timeout_s, exc)
            return ""
        except pytesseract.TesseractError as exc:
            logger.warning("tesseract failed on an image: %s", exc)
            return ""
        return text

    def confidence(self, image: Image) -> tuple[float | None, int]:
        """Return Tesseract's own mean word confidence for ``image``.

        This is a *reference-free* quality signal: unlike CER it needs no
        ground truth, so it is the only quantitative comparison available for
        a page the project did not generate. It reports how sure the recogniser
        is, not how right it is, and the two can disagree -- a confidently
        misread word scores high. It is therefore used to describe arbitrary
        uploads, never to choose a pipeline or to stand in for CER anywhere in
        the evaluation.

        Costs a second Tesseract pass over the image, so callers that also
        need the transcript pay for two recognitions.

        Args:
            image: Grayscale or BGR ``uint8`` image. Not mutated.

        Returns:
            ``(mean_confidence, word_count)`` where the confidence is a
            percentage in ``[0, 100]``, or ``(None, 0)`` when no word was
            recognised or Tesseract failed.

        Raises:
            OCRError: If ``image`` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise OCRError(
                f"TesseractEngine.confidence expected a numpy array, got {type(image)!r}"
            )
        prepared = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        try:
            data = pytesseract.image_to_data(
                prepared,
                lang=self.lang,
                config=self.config,
                timeout=self.timeout_s,
                output_type=pytesseract.Output.DICT,
            )
        except (RuntimeError, pytesseract.TesseractError) as exc:
            logger.warning("tesseract confidence pass failed: %s", exc)
            return None, 0
        # Tesseract emits a row per layout box; only rows carrying a word have
        # a usable confidence, the rest are reported as -1.
        scores = [
            float(conf)
            for conf, word in zip(data.get("conf", ()), data.get("text", ()), strict=False)
            if str(word).strip() and float(conf) >= 0.0
        ]
        if not scores:
            return None, 0
        return fmean(scores), len(scores)
