"""Name-to-class registry for DIP stages, and construction from config.

The registry is the single extension point for image processing: a new stage
becomes available to every pipeline, to ablation and to the stage-montage
figures as soon as it is registered here. Pipelines themselves are never
hard-coded -- they are ordered lists of ``{stage, params}`` entries in the YAML
config, resolved through this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from doculens.dip.base import BaseStage
from doculens.dip.pipeline import Pipeline
from doculens.dip.stages import (
    AdaptiveThresholdStage,
    BilateralFilterStage,
    ClaheStage,
    DeskewStage,
    GrayscaleStage,
    MedianBlurStage,
    MorphologyOpenStage,
    NlMeansDenoiseStage,
    OtsuThresholdStage,
    PerspectiveCorrectStage,
)

STAGE_REGISTRY: dict[str, type[BaseStage]] = {
    GrayscaleStage.name: GrayscaleStage,
    ClaheStage.name: ClaheStage,
    MedianBlurStage.name: MedianBlurStage,
    NlMeansDenoiseStage.name: NlMeansDenoiseStage,
    BilateralFilterStage.name: BilateralFilterStage,
    OtsuThresholdStage.name: OtsuThresholdStage,
    AdaptiveThresholdStage.name: AdaptiveThresholdStage,
    DeskewStage.name: DeskewStage,
    MorphologyOpenStage.name: MorphologyOpenStage,
    PerspectiveCorrectStage.name: PerspectiveCorrectStage,
}
"""Mapping of the stage name used in YAML to its implementing class."""


class UnknownStageError(KeyError):
    """Raised when a config references a stage name that is not registered."""


def available_stages() -> list[str]:
    """Return the sorted names of all registered stages."""
    return sorted(STAGE_REGISTRY)


def register_stage(cls: type[BaseStage]) -> type[BaseStage]:
    """Register ``cls`` under its ``name`` attribute.

    Intended for stages added by future extensions (for example a learned
    enhancement stage) without editing this module.

    Args:
        cls: A :class:`~doculens.dip.base.BaseStage` subclass with a ``name``.

    Returns:
        ``cls`` unchanged, so this can be used as a decorator.

    Raises:
        ValueError: If the name is already taken.
    """
    if cls.name in STAGE_REGISTRY:
        raise ValueError(f"stage name {cls.name!r} is already registered")
    STAGE_REGISTRY[cls.name] = cls
    return cls


def create_stage(name: str, params: Mapping[str, Any] | None = None) -> BaseStage:
    """Instantiate a registered stage.

    Args:
        name: Registered stage name.
        params: Keyword parameters for the stage constructor.

    Returns:
        The constructed stage.

    Raises:
        UnknownStageError: If ``name`` is not registered.
        ValueError: If the parameters are invalid for that stage.
    """
    try:
        cls = STAGE_REGISTRY[name]
    except KeyError:
        raise UnknownStageError(
            f"unknown stage {name!r}; registered stages are {available_stages()}"
        ) from None
    try:
        return cls(**dict(params or {}))
    except TypeError as exc:
        raise ValueError(f"invalid parameters for stage {name!r}: {exc}") from exc


def build_pipeline(name: str, specs: Sequence[Mapping[str, Any]]) -> Pipeline:
    """Build a :class:`~doculens.dip.pipeline.Pipeline` from config entries.

    Args:
        name: Pipeline name, used in results and output paths.
        specs: Ordered entries of the form ``{"stage": str, "params": dict}``.

    Returns:
        The constructed pipeline.

    Raises:
        ValueError: If an entry is malformed or a stage rejects its parameters.
        UnknownStageError: If an entry names an unregistered stage.
    """
    stages: list[BaseStage] = []
    for index, spec in enumerate(specs):
        if "stage" not in spec:
            raise ValueError(f"pipeline {name!r} entry {index} is missing the 'stage' key")
        unexpected = set(spec) - {"stage", "params"}
        if unexpected:
            raise ValueError(
                f"pipeline {name!r} entry {index} has unexpected keys {sorted(unexpected)}; "
                "only 'stage' and 'params' are allowed"
            )
        stages.append(create_stage(str(spec["stage"]), spec.get("params") or {}))
    return Pipeline(name=name, stages=stages)
