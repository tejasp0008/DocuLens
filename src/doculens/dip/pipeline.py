"""Ordered composition of DIP stages.

A :class:`Pipeline` is a pure function from image to image, plus two
conveniences the report needs: capturing every intermediate image, and
producing the leave-one-out variants used for ablation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2

from doculens.dip.base import Image, Stage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageOutput:
    """One intermediate result captured while running a pipeline."""

    index: int
    """Zero-based position of the stage in the pipeline."""
    name: str
    """Stage name."""
    image: Image
    """Output image of that stage."""


class Pipeline:
    """An ordered, immutable list of stages applied left to right."""

    def __init__(self, name: str, stages: Sequence[Stage]) -> None:
        """Initialise the pipeline.

        Args:
            name: Pipeline name as used in config, results and output paths.
            stages: Stages to apply in order. May be empty, which makes the
                pipeline an identity function (useful for ablation of a
                single-stage pipeline).
        """
        self.name = name
        self._stages: tuple[Stage, ...] = tuple(stages)

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Return the stages in application order."""
        return self._stages

    @property
    def stage_names(self) -> list[str]:
        """Return the stage names in application order."""
        return [stage.name for stage in self._stages]

    def __len__(self) -> int:
        """Return the number of stages."""
        return len(self._stages)

    def __iter__(self) -> Iterator[Stage]:
        """Iterate over the stages in application order."""
        return iter(self._stages)

    def __repr__(self) -> str:
        """Return a readable representation listing the stage names."""
        return f"Pipeline(name={self.name!r}, stages={self.stage_names})"

    def __call__(self, image: Image) -> Image:
        """Apply every stage in order and return the final image.

        Args:
            image: Input image. Not mutated.

        Returns:
            The processed image. If the pipeline is empty, the input is
            returned unchanged.
        """
        current = image
        for stage in self._stages:
            current = stage(current)
        return current

    def run_capturing(self, image: Image) -> tuple[Image, list[StageOutput]]:
        """Apply every stage, keeping each intermediate image.

        Args:
            image: Input image. Not mutated.

        Returns:
            A tuple of the final image and the list of per-stage outputs.
        """
        current = image
        captured: list[StageOutput] = []
        for index, stage in enumerate(self._stages):
            current = stage(current)
            captured.append(StageOutput(index=index, name=stage.name, image=current))
        return current, captured

    def run_saving_stages(self, image: Image, out_dir: Path) -> Image:
        """Apply every stage and write the intermediates as PNGs.

        This is the only method on the class with a side effect. Files are named
        ``NN_<stage>.png`` with ``00_input.png`` for the unprocessed image, so
        that lexicographic order is pipeline order.

        Args:
            image: Input image. Not mutated.
            out_dir: Directory to create and write into.

        Returns:
            The final processed image.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / "00_input.png"), image)
        final, captured = self.run_capturing(image)
        for output in captured:
            path = out_dir / f"{output.index + 1:02d}_{output.name}.png"
            cv2.imwrite(str(path), output.image)
        logger.debug("wrote %d stage images to %s", len(captured) + 1, out_dir)
        return final

    def without_stage(self, index: int) -> Pipeline:
        """Return a copy of this pipeline with the stage at ``index`` removed.

        Used by the leave-one-stage-out ablation, which reads the same config
        as the main run so that the two can never drift apart.

        Args:
            index: Zero-based index of the stage to drop.

        Returns:
            A new pipeline named ``<name>__minus_<NN>_<stage>``.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self._stages):
            raise IndexError(f"stage index {index} out of range for pipeline of {len(self)}")
        dropped = self._stages[index]
        remaining = self._stages[:index] + self._stages[index + 1 :]
        return Pipeline(name=f"{self.name}__minus_{index:02d}_{dropped.name}", stages=remaining)

    def describe(self) -> list[dict[str, object]]:
        """Return a JSON-serialisable description of every stage and its params."""
        return [
            {"index": index, "stage": stage.name, "params": stage.params}
            for index, stage in enumerate(self._stages)
        ]
