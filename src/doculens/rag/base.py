"""Policy-rule retrieval interface: a stub only, not implemented in this version.

Claim validation needs the clauses that govern a claim ("is windscreen damage
covered under an own-damage policy?"). That is a retrieval problem over policy
documents, and this protocol is where a vector store, a BM25 index or a hybrid
retriever will attach. See ``docs/EXTENDING.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved passage of policy text.

    Attributes:
        chunk_id: Stable identifier of the passage.
        text: The passage itself.
        score: Retrieval score; higher is more relevant.
        source: Where the passage came from, for citation.
    """

    chunk_id: str
    text: str
    score: float
    source: str


@runtime_checkable
class Retriever(Protocol):
    """Structural type of a policy-rule retriever."""

    name: str

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the passages most relevant to ``query``.

        Args:
            query: Natural-language query.
            top_k: Maximum number of passages to return.

        Returns:
            The retrieved passages, most relevant first.
        """
        ...


class NotImplementedRetriever:
    """Placeholder retriever that documents the interface and refuses to run."""

    name = "not_implemented"

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Raise :class:`NotImplementedError`.

        Args:
            query: Unused.
            top_k: Unused.

        Raises:
            NotImplementedError: Always; retrieval is future work.
        """
        raise NotImplementedError(
            "Policy-rule retrieval is out of scope for this version of DocuLens. "
            "See docs/EXTENDING.md for how to implement Retriever."
        )
