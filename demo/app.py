"""Local demo server for showing the DocuLens pipeline live.

This is a presentation harness, not part of the graded pipeline: it adds no
image-processing logic of its own. Every operation it performs is a call into
``src/doculens`` -- the same degradation registry, the same ``Pipeline``, the
same ``OCREngine`` and the same ``metrics.score`` the evaluation uses. That it
can be written this thinly is the point: pipelines and OCR take plain images
and return plain values, with no filesystem layout baked in, which is exactly
what the future MCP tool layer will need (see ``docs/EXTENDING.md``).

Run it with:

    make demo                      # http://127.0.0.1:8000
    python demo/app.py --port 9000

Requires Flask (``pip install flask``) and a generated corpus (``make data``).
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from doculens.config import BASELINE_METHOD, Config, load_config  # noqa: E402
from doculens.data.corpus import (  # noqa: E402
    CLEAN_CONDITION,
    CLEAN_LEVEL,
    derive_variant_rng,
    read_image,
)
from doculens.data.degradations import apply_degradation  # noqa: E402
from doculens.eval.metrics import normalise_text, score  # noqa: E402
from doculens.logging_setup import configure_logging  # noqa: E402
from doculens.ocr.tesseract_engine import TesseractEngine, ensure_tesseract  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8000
PREVIEW_WIDTH = 760
"""Width in pixels that page images are downscaled to before being sent."""

MAX_UPLOAD_BYTES = 16 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


class DemoState:
    """Holds the objects the request handlers need, built once at startup."""

    def __init__(self, config: Config) -> None:
        """Initialise the demo state.

        Args:
            config: Validated configuration.
        """
        self.config = config
        self.engine = TesseractEngine.from_config(config.ocr.tesseract)
        self.manifest = self._load_manifest(config)

    @staticmethod
    def _load_manifest(config: Config) -> pd.DataFrame:
        """Load the clean-corpus manifest, or return an empty frame.

        Args:
            config: Validated configuration.

        Returns:
            The manifest, empty if the corpus has not been generated yet.
        """
        path = config.paths.manifest_clean
        if not path.is_file():
            logger.warning("no corpus at %s; run `make data` first", path)
            return pd.DataFrame(columns=["doc_id", "doc_type", "split", "png", "txt"])
        return pd.read_csv(path)


state: DemoState | None = None


def _state() -> DemoState:
    """Return the initialised demo state.

    Returns:
        The demo state.

    Raises:
        RuntimeError: If the server was not initialised.
    """
    if state is None:  # pragma: no cover - guarded by main()
        raise RuntimeError("demo state not initialised")
    return state


def encode_png(image: np.ndarray, width: int = PREVIEW_WIDTH) -> str:
    """Downscale an image and encode it as a PNG data URL.

    Args:
        image: Grayscale or BGR image.
        width: Target width in pixels; the aspect ratio is preserved.

    Returns:
        A ``data:image/png;base64,...`` string.

    Raises:
        RuntimeError: If PNG encoding fails.
    """
    height = max(1, int(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, buffer = cv2.imencode(".png", resized)
    if not ok:
        raise RuntimeError("failed to encode preview PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def _read_result(
    image: np.ndarray, ground_truth: str | None, config: Config, engine: TesseractEngine
) -> dict[str, Any]:
    """OCR an image and score it against the ground truth if there is one.

    Args:
        image: Image to recognise.
        ground_truth: Reference text, or ``None`` for an uploaded page.
        config: Validated configuration.
        engine: OCR engine.

    Returns:
        A mapping with the transcript, timings and metrics.
    """
    started = time.perf_counter()
    text = engine.read(image)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result: dict[str, Any] = {
        "text": normalise_text(text, config.eval.normalisation),
        "raw_text": text,
        "ocr_ms": round(elapsed_ms, 1),
        "cer": None,
        "wer": None,
    }
    if ground_truth is not None:
        cer, wer = score(ground_truth, text, config.eval.normalisation)
        result["cer"] = round(cer, 4)
        result["wer"] = round(wer, 4)
    return result


@app.get("/")
def index() -> str:
    """Serve the demo page."""
    return render_template("index.html")


@app.get("/api/options")
def options() -> Any:
    """Return the documents, degradations and pipelines the UI can offer."""
    current = _state()
    config = current.config
    degradations: dict[str, list[int]] = {CLEAN_CONDITION: [CLEAN_LEVEL]}
    for name, degradation in config.degradations.items():
        degradations[name] = sorted(degradation.levels)
    documents = [
        {"doc_id": row.doc_id, "doc_type": row.doc_type, "split": row.split}
        for row in current.manifest.itertuples(index=False)
    ]
    return jsonify(
        {
            "documents": documents,
            "degradations": degradations,
            "pipelines": sorted(config.pipelines),
            "default_pipeline": config.eval.report.headline_method,
            "default_degradation": "lowlight",
            "tesseract_config": config.ocr.tesseract.config,
            "corpus_ready": bool(documents),
        }
    )


def _load_source(payload: dict[str, Any]) -> tuple[np.ndarray, str | None, str]:
    """Resolve the request into an input image and its ground truth.

    Args:
        payload: Parsed request fields.

    Returns:
        A tuple ``(image, ground_truth, label)``. ``ground_truth`` is ``None``
        for an uploaded page, which therefore cannot be scored.

    Raises:
        ValueError: If the requested document or degradation is unknown.
    """
    current = _state()
    config = current.config
    doc_id = str(payload.get("doc_id", ""))
    degradation = str(payload.get("degradation", CLEAN_CONDITION))
    level = int(payload.get("level", CLEAN_LEVEL))

    rows = current.manifest[current.manifest["doc_id"] == doc_id]
    if rows.empty:
        raise ValueError(f"unknown document {doc_id!r}")
    row = rows.iloc[0]
    image = read_image(config.paths.root / row["png"])
    ground_truth = (config.paths.root / row["txt"]).read_text(encoding="utf-8")

    if degradation == CLEAN_CONDITION:
        return image, ground_truth, "clean page"
    if degradation not in config.degradations:
        raise ValueError(f"unknown degradation {degradation!r}")
    params = config.degradations[degradation].params_for(level)
    rng = derive_variant_rng(config.seed, doc_id, degradation, level)
    degraded = apply_degradation(degradation, image, rng, params)
    detail = ", ".join(f"{k}={v}" for k, v in params.items())
    return degraded, ground_truth, f"{degradation} level {level} ({detail})"


def _load_upload() -> tuple[np.ndarray, None, str]:
    """Read an uploaded page from the request.

    Returns:
        A tuple ``(image, None, label)``; uploads have no ground truth.

    Raises:
        ValueError: If no readable image was uploaded.
    """
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        raise ValueError("no file was uploaded")
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode {uploaded.filename!r} as an image")
    return image, None, f"uploaded: {uploaded.filename}"


@app.post("/api/run")
def run() -> Any:
    """Run the baseline and one pipeline over a page and return everything."""
    current = _state()
    config = current.config
    try:
        if request.files:
            image, ground_truth, label = _load_upload()
            pipeline_name = request.form.get("pipeline") or config.eval.report.headline_method
        else:
            payload = request.get_json(silent=True) or {}
            image, ground_truth, label = _load_source(payload)
            pipeline_name = str(payload.get("pipeline") or config.eval.report.headline_method)
        pipeline = config.build_pipeline(pipeline_name)
    except (ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 400

    started = time.perf_counter()
    processed, captured = pipeline.run_capturing(image)
    preprocess_ms = (time.perf_counter() - started) * 1000.0

    baseline = _read_result(image, ground_truth, config, current.engine)
    pipeline_result = _read_result(processed, ground_truth, config, current.engine)
    pipeline_result["preprocess_ms"] = round(preprocess_ms, 1)

    stages = [{"name": "input", "png": encode_png(image)}]
    stages += [
        {
            "name": output.name,
            "png": encode_png(output.image),
            "params": _describe(pipeline, output.index),
        }
        for output in captured
    ]

    return jsonify(
        {
            "label": label,
            "pipeline": pipeline_name,
            "baseline_method": BASELINE_METHOD,
            "stages": stages,
            "baseline": baseline,
            "result": pipeline_result,
            "ground_truth": (
                normalise_text(ground_truth, config.eval.normalisation) if ground_truth else None
            ),
        }
    )


def _describe(pipeline: Any, index: int) -> str:
    """Return a one-line parameter description of a stage.

    Args:
        pipeline: The pipeline the stage belongs to.
        index: Stage index.

    Returns:
        A compact ``key=value`` string, empty when the stage takes no parameters.
    """
    params = pipeline.stages[index].params
    return ", ".join(f"{k}={v}" for k, v in params.items() if k != "foreground_is_dark")


def main() -> int:
    """Parse arguments, check prerequisites and start the server.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    configure_logging()
    ensure_tesseract()

    global state
    state = DemoState(load_config(args.config))
    if state.manifest.empty:
        logger.warning("corpus is empty: run `make data` before demonstrating")

    logger.info("DocuLens demo on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
