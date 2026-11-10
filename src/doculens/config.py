"""Typed configuration: YAML loading, path resolution and validation.

Every tunable value in DocuLens lives in a YAML file (see
``configs/default.yaml``); this module turns that file into frozen dataclasses
and rejects malformed configurations loudly at startup rather than halfway
through a run. In particular, pipelines are validated by actually constructing
their stages, so an unknown stage name or an invalid stage parameter fails
immediately with a message naming the pipeline and the offending entry.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from doculens.dip.pipeline import Pipeline
from doculens.dip.registry import build_pipeline

logger = logging.getLogger(__name__)

BASELINE_METHOD = "baseline_raw"
"""Reserved method name meaning "OCR the image with no preprocessing"."""

REQUIRED_SEVERITY_LEVELS = (1, 2, 3)
"""Every degradation must define exactly these severity levels."""

_SPLIT_TOLERANCE = 1e-6


class ConfigError(ValueError):
    """Raised when a configuration file is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class Paths:
    """Resolved absolute filesystem locations used by the pipeline."""

    root: Path
    data_dir: Path
    clean_dir: Path
    degraded_dir: Path
    outputs_dir: Path
    stages_dir: Path
    results_dir: Path

    @property
    def manifest_clean(self) -> Path:
        """CSV manifest of generated clean documents and their split."""
        return self.data_dir / "manifest_clean.csv"

    @property
    def manifest_degraded(self) -> Path:
        """CSV manifest mapping each degraded file to doc, degradation and level."""
        return self.data_dir / "manifest_degraded.csv"

    @property
    def results_long(self) -> Path:
        """Tidy long-format per-(document, degradation, level, method) results."""
        return self.results_dir / "results_long.csv"

    @property
    def ablation_csv(self) -> Path:
        """Leave-one-stage-out ablation results."""
        return self.results_dir / "ablation.csv"

    @property
    def figures_dir(self) -> Path:
        """Directory holding generated charts and figures."""
        return self.results_dir / "figures"

    @property
    def failures_dir(self) -> Path:
        """Directory holding worst-case side-by-side figures."""
        return self.results_dir / "failures"

    def ensure_dirs(self) -> None:
        """Create every output directory, including the ``.gitkeep`` parents."""
        for path in (
            self.data_dir,
            self.clean_dir,
            self.degraded_dir,
            self.outputs_dir,
            self.stages_dir,
            self.results_dir,
            self.figures_dir,
            self.failures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CanvasConfig:
    """Page canvas geometry for synthetic document rendering."""

    width: int
    height: int
    background: int


@dataclass(frozen=True)
class FontConfig:
    """Font search paths and sizes for synthetic document rendering."""

    candidates: tuple[str, ...]
    bold_candidates: tuple[str, ...]
    size_title: int
    size_body: int


@dataclass(frozen=True)
class SplitConfig:
    """Train/val/test fractions, fixed before any tuning."""

    train: float
    val: float
    test: float

    def validate(self) -> None:
        """Check the fractions are positive and sum to one.

        Raises:
            ConfigError: If any fraction is non-positive or they do not sum to 1.
        """
        for name, value in (("train", self.train), ("val", self.val), ("test", self.test)):
            if value <= 0:
                raise ConfigError(f"data.split.{name} must be > 0, got {value!r}")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > _SPLIT_TOLERANCE:
            raise ConfigError(f"data.split fractions must sum to 1.0, got {total!r}")


@dataclass(frozen=True)
class DataConfig:
    """Synthetic corpus generation settings."""

    n_docs: int
    locale: str
    doc_types: tuple[str, ...]
    canvas: CanvasConfig
    margin: int
    line_spacing: int
    text_colour: int
    font: FontConfig
    split: SplitConfig


@dataclass(frozen=True)
class DegradationConfig:
    """One registered degradation and its three severity levels."""

    name: str
    levels: dict[int, dict[str, Any]]

    def params_for(self, level: int) -> dict[str, Any]:
        """Return the parameters for a severity level.

        Args:
            level: Severity level.

        Returns:
            The parameter mapping for that level.

        Raises:
            ConfigError: If the level is not configured.
        """
        try:
            return dict(self.levels[level])
        except KeyError:
            raise ConfigError(
                f"degradation {self.name!r} has no level {level}; "
                f"configured levels are {sorted(self.levels)}"
            ) from None


@dataclass(frozen=True)
class TesseractConfig:
    """Tesseract invocation settings."""

    lang: str
    config: str
    timeout_s: int


@dataclass(frozen=True)
class OcrConfig:
    """OCR engine selection and engine-specific settings."""

    engine: str
    tesseract: TesseractConfig


@dataclass(frozen=True)
class SegmentationConfig:
    """Connected-component field-region segmentation settings."""

    enabled: bool
    dilate_kernel_width: int
    dilate_kernel_height: int
    min_region_width_frac: float
    min_region_height_frac: float
    max_region_height_frac: float


@dataclass(frozen=True)
class NormalisationConfig:
    """Text normalisation applied before computing CER and WER."""

    lowercase: bool
    collapse_whitespace: bool
    strip: bool
    remove_punctuation: bool


@dataclass(frozen=True)
class AblationConfig:
    """Which pipeline the leave-one-stage-out ablation operates on."""

    pipeline: str


@dataclass(frozen=True)
class FailuresConfig:
    """Worst-case selection settings."""

    top_k: int
    method: str


@dataclass(frozen=True)
class ReportConfig:
    """Report asset generation settings."""

    dpi: int
    stage_sample_docs: int
    stage_sample_degradation: str
    stage_sample_level: int
    headline_method: str


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation, ablation, failure-selection and reporting settings."""

    normalisation: NormalisationConfig
    methods: tuple[str, ...]
    report_splits: tuple[str, ...]
    ablation: AblationConfig
    failures: FailuresConfig
    report: ReportConfig


@dataclass(frozen=True)
class Config:
    """The fully validated DocuLens configuration."""

    seed: int
    paths: Paths
    data: DataConfig
    degradations: dict[str, DegradationConfig]
    pipelines: dict[str, tuple[dict[str, Any], ...]]
    ocr: OcrConfig
    segmentation: SegmentationConfig
    eval: EvalConfig
    source_path: Path = field(default_factory=lambda: Path("configs/default.yaml"))

    def build_pipeline(self, name: str) -> Pipeline:
        """Construct a named pipeline from the config.

        Args:
            name: A key of :attr:`pipelines`.

        Returns:
            The constructed pipeline.

        Raises:
            ConfigError: If no pipeline by that name is configured.
        """
        if name not in self.pipelines:
            raise ConfigError(
                f"unknown pipeline {name!r}; configured pipelines are {sorted(self.pipelines)}"
            )
        return build_pipeline(name, self.pipelines[name])

    @property
    def pipeline_methods(self) -> tuple[str, ...]:
        """Return the evaluated methods that are pipelines (excluding the baseline)."""
        return tuple(m for m in self.eval.methods if m != BASELINE_METHOD)

    def seed_everything(self) -> None:
        """Seed ``random`` and ``numpy`` from :attr:`seed` for reproducibility.

        Faker is seeded separately in the generator, because its seeding is
        per-instance.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        logger.debug("seeded random and numpy with %d", self.seed)


def _require_mapping(raw: Any, where: str) -> dict[str, Any]:
    """Return ``raw`` as a dict, or raise :class:`ConfigError`.

    Args:
        raw: Value to check.
        where: Dotted config path, used in the error message.

    Returns:
        The value as a plain dict.

    Raises:
        ConfigError: If ``raw`` is not a mapping.
    """
    if not isinstance(raw, Mapping):
        raise ConfigError(f"config section {where!r} must be a mapping, got {type(raw).__name__}")
    return dict(raw)


def _require_key(raw: Mapping[str, Any], key: str, where: str) -> Any:
    """Return ``raw[key]``, or raise :class:`ConfigError` naming the location.

    Args:
        raw: Mapping to read from.
        key: Required key.
        where: Dotted config path, used in the error message.

    Returns:
        The value at ``key``.

    Raises:
        ConfigError: If the key is absent.
    """
    if key not in raw:
        raise ConfigError(f"config section {where!r} is missing the required key {key!r}")
    return raw[key]


def _resolve(root: Path, value: str) -> Path:
    """Resolve a config path against the project root.

    Args:
        root: Project root directory.
        value: Absolute or root-relative path string.

    Returns:
        An absolute path.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path)


def _project_root(config_path: Path) -> Path:
    """Infer the project root from the config file location.

    A config at ``<root>/configs/default.yaml`` yields ``<root>``; any other
    location yields the config file's own directory.

    Args:
        config_path: Path to the YAML config.

    Returns:
        The inferred project root.
    """
    parent = config_path.parent
    return parent.parent if parent.name == "configs" else parent


def load_config(path: str | Path) -> Config:
    """Load, resolve and validate a YAML configuration file.

    Args:
        path: Path to the YAML config file.

    Returns:
        The validated :class:`Config`.

    Raises:
        ConfigError: If the file is missing, unparseable or inconsistent, or if
            it references an unknown stage, pipeline or degradation.
    """
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse {config_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{config_path} must contain a top-level mapping")

    root = _project_root(config_path.resolve())
    config = _build_config(raw, root, config_path.resolve())
    _validate(config)
    logger.debug("loaded config from %s (root=%s)", config_path, root)
    return config


def _build_config(raw: Mapping[str, Any], root: Path, source: Path) -> Config:
    """Convert a raw mapping into a :class:`Config` without cross-validation.

    Args:
        raw: Parsed YAML mapping.
        root: Project root for path resolution.
        source: Path of the config file, retained for provenance.

    Returns:
        The populated config.

    Raises:
        ConfigError: If a required section or key is missing or mistyped.
    """
    seed = _require_key(raw, "seed", "<top level>")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ConfigError(f"seed must be an integer, got {seed!r}")

    paths_raw = _require_mapping(_require_key(raw, "paths", "<top level>"), "paths")
    paths = Paths(
        root=root,
        data_dir=_resolve(root, _require_key(paths_raw, "data_dir", "paths")),
        clean_dir=_resolve(root, _require_key(paths_raw, "clean_dir", "paths")),
        degraded_dir=_resolve(root, _require_key(paths_raw, "degraded_dir", "paths")),
        outputs_dir=_resolve(root, _require_key(paths_raw, "outputs_dir", "paths")),
        stages_dir=_resolve(root, _require_key(paths_raw, "stages_dir", "paths")),
        results_dir=_resolve(root, _require_key(paths_raw, "results_dir", "paths")),
    )

    data_raw = _require_mapping(_require_key(raw, "data", "<top level>"), "data")
    canvas_raw = _require_mapping(_require_key(data_raw, "canvas", "data"), "data.canvas")
    font_raw = _require_mapping(_require_key(data_raw, "font", "data"), "data.font")
    split_raw = _require_mapping(_require_key(data_raw, "split", "data"), "data.split")
    data = DataConfig(
        n_docs=int(_require_key(data_raw, "n_docs", "data")),
        locale=str(_require_key(data_raw, "locale", "data")),
        doc_types=tuple(str(t) for t in _require_key(data_raw, "doc_types", "data")),
        canvas=CanvasConfig(
            width=int(_require_key(canvas_raw, "width", "data.canvas")),
            height=int(_require_key(canvas_raw, "height", "data.canvas")),
            background=int(_require_key(canvas_raw, "background", "data.canvas")),
        ),
        margin=int(_require_key(data_raw, "margin", "data")),
        line_spacing=int(_require_key(data_raw, "line_spacing", "data")),
        text_colour=int(_require_key(data_raw, "text_colour", "data")),
        font=FontConfig(
            candidates=tuple(str(p) for p in _require_key(font_raw, "candidates", "data.font")),
            bold_candidates=tuple(
                str(p) for p in _require_key(font_raw, "bold_candidates", "data.font")
            ),
            size_title=int(_require_key(font_raw, "size_title", "data.font")),
            size_body=int(_require_key(font_raw, "size_body", "data.font")),
        ),
        split=SplitConfig(
            train=float(_require_key(split_raw, "train", "data.split")),
            val=float(_require_key(split_raw, "val", "data.split")),
            test=float(_require_key(split_raw, "test", "data.split")),
        ),
    )

    degradations_raw = _require_mapping(
        _require_key(raw, "degradations", "<top level>"), "degradations"
    )
    degradations: dict[str, DegradationConfig] = {}
    for name, spec in degradations_raw.items():
        spec_map = _require_mapping(spec, f"degradations.{name}")
        levels_raw = _require_mapping(
            _require_key(spec_map, "levels", f"degradations.{name}"),
            f"degradations.{name}.levels",
        )
        levels = {
            int(level): dict(_require_mapping(params, f"degradations.{name}.levels.{level}"))
            for level, params in levels_raw.items()
        }
        degradations[str(name)] = DegradationConfig(name=str(name), levels=levels)

    pipelines_raw = _require_mapping(_require_key(raw, "pipelines", "<top level>"), "pipelines")
    pipelines: dict[str, tuple[dict[str, Any], ...]] = {}
    for name, specs in pipelines_raw.items():
        if not isinstance(specs, Sequence) or isinstance(specs, str):
            raise ConfigError(f"pipelines.{name} must be a list of stage entries")
        pipelines[str(name)] = tuple(
            dict(_require_mapping(spec, f"pipelines.{name}[{i}]")) for i, spec in enumerate(specs)
        )

    ocr_raw = _require_mapping(_require_key(raw, "ocr", "<top level>"), "ocr")
    tess_raw = _require_mapping(_require_key(ocr_raw, "tesseract", "ocr"), "ocr.tesseract")
    ocr = OcrConfig(
        engine=str(_require_key(ocr_raw, "engine", "ocr")),
        tesseract=TesseractConfig(
            lang=str(_require_key(tess_raw, "lang", "ocr.tesseract")),
            config=str(_require_key(tess_raw, "config", "ocr.tesseract")),
            timeout_s=int(_require_key(tess_raw, "timeout_s", "ocr.tesseract")),
        ),
    )

    seg_raw = _require_mapping(_require_key(raw, "segmentation", "<top level>"), "segmentation")
    segmentation = SegmentationConfig(
        enabled=bool(_require_key(seg_raw, "enabled", "segmentation")),
        dilate_kernel_width=int(_require_key(seg_raw, "dilate_kernel_width", "segmentation")),
        dilate_kernel_height=int(_require_key(seg_raw, "dilate_kernel_height", "segmentation")),
        min_region_width_frac=float(_require_key(seg_raw, "min_region_width_frac", "segmentation")),
        min_region_height_frac=float(
            _require_key(seg_raw, "min_region_height_frac", "segmentation")
        ),
        max_region_height_frac=float(
            _require_key(seg_raw, "max_region_height_frac", "segmentation")
        ),
    )

    eval_raw = _require_mapping(_require_key(raw, "eval", "<top level>"), "eval")
    norm_raw = _require_mapping(
        _require_key(eval_raw, "normalisation", "eval"), "eval.normalisation"
    )
    abl_raw = _require_mapping(_require_key(eval_raw, "ablation", "eval"), "eval.ablation")
    fail_raw = _require_mapping(_require_key(eval_raw, "failures", "eval"), "eval.failures")
    rep_raw = _require_mapping(_require_key(eval_raw, "report", "eval"), "eval.report")
    eval_cfg = EvalConfig(
        normalisation=NormalisationConfig(
            lowercase=bool(_require_key(norm_raw, "lowercase", "eval.normalisation")),
            collapse_whitespace=bool(
                _require_key(norm_raw, "collapse_whitespace", "eval.normalisation")
            ),
            strip=bool(_require_key(norm_raw, "strip", "eval.normalisation")),
            remove_punctuation=bool(
                _require_key(norm_raw, "remove_punctuation", "eval.normalisation")
            ),
        ),
        methods=tuple(str(m) for m in _require_key(eval_raw, "methods", "eval")),
        report_splits=tuple(str(s) for s in _require_key(eval_raw, "report_splits", "eval")),
        ablation=AblationConfig(pipeline=str(_require_key(abl_raw, "pipeline", "eval.ablation"))),
        failures=FailuresConfig(
            top_k=int(_require_key(fail_raw, "top_k", "eval.failures")),
            method=str(_require_key(fail_raw, "method", "eval.failures")),
        ),
        report=ReportConfig(
            dpi=int(_require_key(rep_raw, "dpi", "eval.report")),
            stage_sample_docs=int(_require_key(rep_raw, "stage_sample_docs", "eval.report")),
            stage_sample_degradation=str(
                _require_key(rep_raw, "stage_sample_degradation", "eval.report")
            ),
            stage_sample_level=int(_require_key(rep_raw, "stage_sample_level", "eval.report")),
            headline_method=str(_require_key(rep_raw, "headline_method", "eval.report")),
        ),
    )

    return Config(
        seed=seed,
        paths=paths,
        data=data,
        degradations=degradations,
        pipelines=pipelines,
        ocr=ocr,
        segmentation=segmentation,
        eval=eval_cfg,
        source_path=source,
    )


def _validate(config: Config) -> None:
    """Cross-validate a built config.

    Checks corpus sizes, split fractions, severity levels, pipeline stage names
    and parameters (by constructing them), and that every evaluated method,
    ablation target, failure method and headline method resolves.

    Args:
        config: The config to validate.

    Raises:
        ConfigError: On the first inconsistency found.
    """
    if config.data.n_docs < 1:
        raise ConfigError(f"data.n_docs must be >= 1, got {config.data.n_docs}")
    if not config.data.doc_types:
        raise ConfigError("data.doc_types must list at least one document type")
    if config.data.canvas.width < 1 or config.data.canvas.height < 1:
        raise ConfigError("data.canvas width and height must be positive")
    config.data.split.validate()

    if not config.degradations:
        raise ConfigError("at least one degradation must be configured")
    from doculens.data.degradations import DEGRADATION_REGISTRY

    for name, degradation in config.degradations.items():
        if name not in DEGRADATION_REGISTRY:
            raise ConfigError(
                f"unknown degradation {name!r}; registered degradations are "
                f"{sorted(DEGRADATION_REGISTRY)}"
            )
        missing = set(REQUIRED_SEVERITY_LEVELS) - set(degradation.levels)
        extra = set(degradation.levels) - set(REQUIRED_SEVERITY_LEVELS)
        if missing or extra:
            raise ConfigError(
                f"degradation {name!r} must define exactly levels "
                f"{list(REQUIRED_SEVERITY_LEVELS)}, got {sorted(degradation.levels)}"
            )

    if not config.pipelines:
        raise ConfigError("at least one pipeline must be configured")
    for name in config.pipelines:
        try:
            config.build_pipeline(name)
        except (ValueError, KeyError) as exc:
            raise ConfigError(f"invalid pipeline {name!r}: {exc}") from exc

    if not config.eval.methods:
        raise ConfigError("eval.methods must list at least one method")
    for method in config.eval.methods:
        if method != BASELINE_METHOD and method not in config.pipelines:
            raise ConfigError(
                f"eval.methods references {method!r}, which is neither {BASELINE_METHOD!r} "
                f"nor a configured pipeline ({sorted(config.pipelines)})"
            )
    valid_splits = {"train", "val", "test"}
    for split in config.eval.report_splits:
        if split not in valid_splits:
            raise ConfigError(f"eval.report_splits contains unknown split {split!r}")
    if config.eval.ablation.pipeline not in config.pipelines:
        raise ConfigError(
            f"eval.ablation.pipeline {config.eval.ablation.pipeline!r} is not a configured pipeline"
        )
    if config.eval.failures.method not in config.eval.methods:
        raise ConfigError(
            f"eval.failures.method {config.eval.failures.method!r} is not in eval.methods"
        )
    if config.eval.failures.top_k < 1:
        raise ConfigError("eval.failures.top_k must be >= 1")
    if config.eval.report.headline_method not in config.eval.methods:
        raise ConfigError(
            f"eval.report.headline_method {config.eval.report.headline_method!r} "
            "is not in eval.methods"
        )
    if config.eval.report.dpi < 1:
        raise ConfigError("eval.report.dpi must be >= 1")
    if config.eval.report.stage_sample_degradation not in config.degradations:
        raise ConfigError(
            f"eval.report.stage_sample_degradation "
            f"{config.eval.report.stage_sample_degradation!r} is not a configured degradation"
        )
    if config.ocr.engine != "tesseract":
        raise ConfigError(
            f"ocr.engine {config.ocr.engine!r} is not implemented; only 'tesseract' is "
            "available in this version (see docs/EXTENDING.md to add an engine)"
        )
