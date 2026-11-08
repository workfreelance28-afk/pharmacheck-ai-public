"""Shared contract for replaceable medical interaction sources."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from pharmacheck.contracts import EvidenceDocument, NonEmptyText, SearchResult, SourceId


class InteractionSearchOutcome(BaseModel):
    """Evidence and coverage metadata returned for one ingredient pair."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    provider_id: SourceId
    ingredient_pair: tuple[NonEmptyText, NonEmptyText]
    search_result: SearchResult
    documents: list[EvidenceDocument] = Field(default_factory=list)
    complete: bool
    failed_queries: list[str] = Field(default_factory=list)


@runtime_checkable
class InteractionSource(Protocol):
    """Provider-neutral interface used by the future evidence orchestrator."""

    def search_pair(
        self,
        ingredient_a: str,
        ingredient_b: str,
        *,
        checked_at: datetime | None = None,
    ) -> InteractionSearchOutcome: ...
