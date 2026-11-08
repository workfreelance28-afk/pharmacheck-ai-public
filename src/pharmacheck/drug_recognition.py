"""Safe parsing and catalog-based drug-name resolution for stage 2."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from difflib import SequenceMatcher
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pharmacheck.contracts import Confidence, IdentityStatus, NonEmptyText, SourceId


class RecognitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PairParseStatus(StrEnum):
    PARSED = "parsed"
    INCOMPLETE = "incomplete"
    TOO_MANY = "too_many"


class MentionedDrug(RecognitionModel):
    raw_text: NonEmptyText
    name: NonEmptyText
    normalized_name: NonEmptyText
    strength: str | None = None
    dosage_form: str | None = None


class ParsedDrugPair(RecognitionModel):
    status: PairParseStatus
    drugs: list[MentionedDrug] = Field(default_factory=list, max_length=2)
    clarification: str | None = None
    # True when the pair was taken out of a free-text phrase; the rest of the
    # message may describe symptoms, so it must reach YandexGPT (D-018).
    free_text: bool = False

    @model_validator(mode="after")
    def parsed_requires_two_drugs(self) -> ParsedDrugPair:
        if self.status is PairParseStatus.PARSED and len(self.drugs) != 2:
            raise ValueError("parsed pair requires exactly two drugs")
        return self


class DrugCatalogEntry(RecognitionModel):
    product_id: SourceId
    trade_name: NonEmptyText
    aliases: list[NonEmptyText] = Field(default_factory=list)
    dosage_form: str | None = None
    strength: str | None = None
    active_ingredients: list[NonEmptyText] = Field(min_length=1)
    source_id: SourceId
    registration_numbers: list[NonEmptyText] = Field(default_factory=list)


class CandidateDrug(RecognitionModel):
    product_id: SourceId
    trade_name: NonEmptyText
    dosage_form: str | None
    strength: str | None
    active_ingredients: list[NonEmptyText]
    source_id: SourceId
    similarity: float = Field(ge=0, le=1)


class DrugResolution(RecognitionModel):
    mention: MentionedDrug
    identity_status: IdentityStatus
    identity_confidence: Confidence
    normalized_trade_name: str | None = None
    dosage_form: str | None = None
    strength: str | None = None
    active_ingredients: list[NonEmptyText] = Field(default_factory=list)
    identity_source_ids: list[SourceId] = Field(default_factory=list)
    candidates: list[CandidateDrug] = Field(default_factory=list, max_length=3)
    clarification: str | None = None


class PairResolutionStatus(StrEnum):
    READY = "ready"
    CLARIFICATION_NEEDED = "clarification_needed"


class ResolvedDrugPair(RecognitionModel):
    status: PairResolutionStatus
    drugs: list[DrugResolution] = Field(max_length=2)
    clarification: str | None = None


FORM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("таблетки", re.compile(r"\b(?:таблетк(?:а|и|у|ах)?|таб\.?|tbl\.?)\b", re.I)),
    ("капсулы", re.compile(r"\b(?:капсул(?:а|ы|у|ах)?|кап\.?)\b", re.I)),
    ("раствор", re.compile(r"\b(?:раствор(?:а|е)?|р-р)\b", re.I)),
    ("сироп", re.compile(r"\bсироп(?:а|е)?\b", re.I)),
    ("спрей", re.compile(r"\bспрей\b", re.I)),
    ("мазь", re.compile(r"\bмаз(?:ь|и)\b", re.I)),
    ("крем", re.compile(r"\bкрем(?:а|е)?\b", re.I)),
    ("капли", re.compile(r"\bкап(?:ли|ель|лях)\b", re.I)),
)

STRENGTH_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?\s*(?:мкг|мг|г|мл|%))(?!\w)",
    re.IGNORECASE,
)

LABEL_RE = re.compile(r"^\s*препарат\s*[12]\s*:\s*", re.IGNORECASE)
TAKES_RE = re.compile(
    r"(?:я\s+)?принимаю\s+(.+?)(?:,\s*а\s+|[.;,\n]|\s+и\s+)\s*(?:мне\s+)?"
    r"назначил(?:и|а)?\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)

SUGGESTION_SIMILARITY_THRESHOLD = 0.78
LONG_NAME_MIN_LENGTH = 6
LONG_NAME_PREFILTER_THRESHOLD = 0.68
MAX_LONG_NAME_EDIT_DISTANCE = 2


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    normalized = re.sub(r"[^\w\s-]", " ", normalized, flags=re.UNICODE)
    normalized = normalized.replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def normalize_strength(value: str | None) -> str | None:
    if not value:
        return None
    return "".join(value.casefold().replace(",", ".").split())


def parse_drug_pair(message: str) -> ParsedDrugPair:
    text = unicodedata.normalize("NFKC", message).strip()
    text = re.sub(r"^/check(?:@\w+)?\s*", "", text, flags=re.IGNORECASE)
    if not text:
        return _incomplete_pair()

    phrase_match = TAKES_RE.search(text)
    free_text = phrase_match is not None
    if phrase_match:
        parts = [phrase_match.group(1), phrase_match.group(2)]
    elif "+" in text:
        parts = [part.strip() for part in text.split("+") if part.strip()]
        if len(parts) > 2:
            return ParsedDrugPair(
                status=PairParseStatus.TOO_MANY,
                clarification="Для MVP укажите ровно два препарата.",
            )
    else:
        lines = [LABEL_RE.sub("", line).strip() for line in text.splitlines()]
        parts = [line for line in lines if line]

    if len(parts) != 2:
        return _incomplete_pair()

    mentions = [_parse_mention(part) for part in parts]
    if any(not mention.normalized_name for mention in mentions):
        return _incomplete_pair()
    return ParsedDrugPair(
        status=PairParseStatus.PARSED,
        drugs=mentions,
        free_text=free_text,
    )


def _incomplete_pair() -> ParsedDrugPair:
    return ParsedDrugPair(
        status=PairParseStatus.INCOMPLETE,
        clarification="Укажите два препарата, например: «Препарат А + Препарат Б».",
    )


def _parse_mention(raw_text: str) -> MentionedDrug:
    cleaned = raw_text.strip().strip("\"'«»„“” ")
    strength_match = STRENGTH_RE.search(cleaned)
    strength = normalize_strength(strength_match.group(1)) if strength_match else None

    dosage_form = None
    name_without_metadata = cleaned
    if strength_match:
        name_without_metadata = STRENGTH_RE.sub(" ", name_without_metadata, count=1)
    for canonical_form, pattern in FORM_PATTERNS:
        if pattern.search(name_without_metadata):
            dosage_form = canonical_form
            name_without_metadata = pattern.sub(" ", name_without_metadata, count=1)
            break

    name = " ".join(name_without_metadata.strip(" ,.;:()[]").split())
    return MentionedDrug(
        raw_text=cleaned,
        name=name,
        normalized_name=normalize_name(name),
        strength=strength,
        dosage_form=dosage_form,
    )


class DrugCatalog:
    """Immutable in-memory index; stage 3 can feed it from an official source."""

    def __init__(self, entries: Iterable[DrugCatalogEntry]) -> None:
        self._entries = tuple(entries)
        if not self._entries:
            raise ValueError("drug catalog cannot be empty")
        self._aliases: dict[str, list[DrugCatalogEntry]] = {}
        for entry in self._entries:
            normalized_names = {
                normalize_name(name) for name in {entry.trade_name, *entry.aliases}
            }
            for normalized in normalized_names:
                self._aliases.setdefault(normalized, []).append(entry)

    @classmethod
    def from_json(cls, path: str | Path) -> DrugCatalog:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(DrugCatalogEntry.model_validate(item) for item in raw)

    def resolve(self, mention: MentionedDrug) -> DrugResolution:
        exact = list(self._aliases.get(mention.normalized_name, ()))
        filtered = self._filter_by_metadata(exact, mention)

        if len(filtered) == 1:
            return self._resolved(mention, filtered[0])
        if len(filtered) > 1 or exact:
            candidates = filtered if filtered else exact
            return DrugResolution(
                mention=mention,
                identity_status=IdentityStatus.AMBIGUOUS,
                identity_confidence=Confidence.LOW,
                candidates=self._candidate_list(candidates, mention.normalized_name),
                clarification=(
                    "Название соответствует нескольким препаратам. "
                    "Уточните лекарственную форму и дозировку."
                ),
            )

        suggestions = self._suggestions(mention.normalized_name)
        return DrugResolution(
            mention=mention,
            identity_status=IdentityStatus.UNRESOLVED,
            identity_confidence=Confidence.LOW,
            candidates=suggestions,
            clarification=(
                "Точное совпадение не найдено. Подтвердите одно из похожих названий."
                if suggestions
                else "Название не найдено в каталоге. Проверьте написание."
            ),
        )

    def resolve_pair(self, parsed: ParsedDrugPair) -> ResolvedDrugPair:
        if parsed.status is not PairParseStatus.PARSED:
            return ResolvedDrugPair(
                status=PairResolutionStatus.CLARIFICATION_NEEDED,
                drugs=[],
                clarification=parsed.clarification,
            )
        resolutions = [self.resolve(mention) for mention in parsed.drugs]
        ready = all(
            resolution.identity_status is IdentityStatus.RESOLVED
            and resolution.identity_confidence is Confidence.HIGH
            for resolution in resolutions
        )
        clarification = None
        if not ready:
            messages = [
                resolution.clarification
                for resolution in resolutions
                if resolution.clarification
            ]
            clarification = " ".join(dict.fromkeys(messages))
        return ResolvedDrugPair(
            status=(
                PairResolutionStatus.READY
                if ready
                else PairResolutionStatus.CLARIFICATION_NEEDED
            ),
            drugs=resolutions,
            clarification=clarification,
        )

    def _filter_by_metadata(
        self,
        entries: list[DrugCatalogEntry],
        mention: MentionedDrug,
    ) -> list[DrugCatalogEntry]:
        filtered = entries
        if mention.dosage_form:
            form = normalize_name(mention.dosage_form)
            filtered = [
                entry
                for entry in filtered
                if entry.dosage_form is None or normalize_name(entry.dosage_form) == form
            ]
        if mention.strength:
            strength = normalize_strength(mention.strength)
            filtered = [
                entry
                for entry in filtered
                if entry.strength is None or normalize_strength(entry.strength) == strength
            ]
        return filtered

    def _suggestions(self, normalized_name: str) -> list[CandidateDrug]:
        scored: dict[str, tuple[float, DrugCatalogEntry]] = {}
        for alias, entries in self._aliases.items():
            score = SequenceMatcher(None, normalized_name, alias).ratio()
            if not _is_suggestion_candidate(normalized_name, alias, score):
                continue
            for entry in entries:
                current = scored.get(entry.product_id)
                if current is None or score > current[0]:
                    scored[entry.product_id] = (score, entry)
        ordered = sorted(scored.values(), key=lambda item: (-item[0], item[1].trade_name))
        return [self._candidate(entry, score) for score, entry in ordered[:3]]

    def _candidate_list(
        self,
        entries: list[DrugCatalogEntry],
        normalized_name: str,
    ) -> list[CandidateDrug]:
        return [
            self._candidate(
                entry,
                max(
                    SequenceMatcher(None, normalized_name, normalize_name(name)).ratio()
                    for name in {entry.trade_name, *entry.aliases}
                ),
            )
            for entry in entries[:3]
        ]

    @staticmethod
    def _candidate(entry: DrugCatalogEntry, score: float) -> CandidateDrug:
        return CandidateDrug(
            product_id=entry.product_id,
            trade_name=entry.trade_name,
            dosage_form=entry.dosage_form,
            strength=entry.strength,
            active_ingredients=entry.active_ingredients,
            source_id=entry.source_id,
            similarity=round(score, 4),
        )

    @staticmethod
    def _resolved(mention: MentionedDrug, entry: DrugCatalogEntry) -> DrugResolution:
        return DrugResolution(
            mention=mention,
            identity_status=IdentityStatus.RESOLVED,
            identity_confidence=Confidence.HIGH,
            normalized_trade_name=entry.trade_name,
            dosage_form=entry.dosage_form,
            strength=entry.strength,
            active_ingredients=entry.active_ingredients,
            identity_source_ids=[entry.source_id],
        )


def _is_suggestion_candidate(query: str, alias: str, similarity: float) -> bool:
    """Allow wider typo suggestions for long names without auto-resolving them."""

    if similarity >= SUGGESTION_SIMILARITY_THRESHOLD:
        return True
    if (
        len(query) < LONG_NAME_MIN_LENGTH
        or len(alias) < LONG_NAME_MIN_LENGTH
        or similarity < LONG_NAME_PREFILTER_THRESHOLD
        or query[0] != alias[0]
        or abs(len(query) - len(alias)) > MAX_LONG_NAME_EDIT_DISTANCE
    ):
        return False
    return _edit_distance(query, alias) <= MAX_LONG_NAME_EDIT_DISTANCE


def _edit_distance(left: str, right: str) -> int:
    """Return the Levenshtein distance using one row of memory."""

    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]
