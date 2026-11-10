"""Tool-service interface: a stub only, not implemented in this version.

The planned MCP server does not reimplement anything: each tool it exposes is a
thin wrapper over a function that already exists and is already tested here.

====================  ==================================================
Planned MCP tool      Wraps
====================  ==================================================
``enhance_image``     :meth:`doculens.dip.pipeline.Pipeline.__call__`
``run_ocr``           :meth:`doculens.ocr.base.OCREngine.read`
``classify_document`` future CNN/ViT classifier (not started)
``extract_fields``    :meth:`doculens.extraction.base.FieldExtractor.extract`
``search_policy_rules`` :meth:`doculens.rag.base.Retriever.search`
``validate_claim``    future agent loop over the four tools above
====================  ==================================================

This is why pipeline and OCR functions are pure and take plain images rather
than paths: a tool call has no filesystem layout to rely on. See
``docs/EXTENDING.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSpec:
    """Declaration of one callable tool.

    Attributes:
        name: Tool name as exposed to the caller.
        description: One-line description for the tool catalogue.
        input_schema: JSON-schema fragment describing the arguments.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolService(Protocol):
    """Structural type of a service exposing DocuLens functionality as tools."""

    name: str

    def list_tools(self) -> list[ToolSpec]:
        """Return the tools this service exposes."""
        ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke one tool by name.

        Args:
            name: Tool name from :meth:`list_tools`.
            arguments: Tool arguments matching its input schema.

        Returns:
            The tool result as a JSON-serialisable mapping.
        """
        ...


PLANNED_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("enhance_image", "Run a named DIP pipeline over an image and return the result."),
    ToolSpec("run_ocr", "Recognise text in an image with the configured OCR engine."),
    ToolSpec("classify_document", "Predict the document type (claim form, invoice, ...)."),
    ToolSpec("extract_fields", "Recover structured claim fields from a page and transcript."),
    ToolSpec("search_policy_rules", "Retrieve policy clauses relevant to a query."),
    ToolSpec("validate_claim", "Check a claim against retrieved policy rules."),
)
"""The tool surface the future MCP server will expose, for reference only."""


class NotImplementedToolService:
    """Placeholder service that documents the planned tools and refuses to run."""

    name = "not_implemented"

    def list_tools(self) -> list[ToolSpec]:
        """Return the planned tool specifications.

        Returns:
            The tools the future MCP server is intended to expose.
        """
        return list(PLANNED_TOOLS)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Raise :class:`NotImplementedError`.

        Args:
            name: Unused.
            arguments: Unused.

        Raises:
            NotImplementedError: Always; the tool service is future work.
        """
        raise NotImplementedError(
            "The MCP tool service is out of scope for this version of DocuLens. "
            "See docs/EXTENDING.md for how each tool maps onto existing functions."
        )
