"""Conservative adapter for the official openFDA Drug Label API."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pharmacheck.contracts import (
    EvidenceDocument,
    RetrievalStatus,
    SearchResult,
    SearchResultStatus,
)
from pharmacheck.sources.base import InteractionSearchOutcome

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_RELAY_LABEL_URL = (
    "https://pharmacheck-openfda-relay.your-account.workers.dev/drug/label.json"
)
APPROVED_OPENFDA_LABEL_URLS = frozenset(
    {OPENFDA_LABEL_URL, OPENFDA_RELAY_LABEL_URL}
)
OPENFDA_RELAY_SECRET_HEADER = "X-PharmaCheck-Relay-Secret"
PUBLISHER = "U.S. Food and Drug Administration (openFDA)"
HUMAN_PRODUCT_TYPES = (
    "HUMAN PRESCRIPTION DRUG",
    "HUMAN OTC DRUG",
)
RELEVANT_SECTIONS = (
    "drug_interactions",
    "drug_and_or_laboratory_test_interactions",
    "contraindications",
    "boxed_warning",
    "warnings_and_cautions",
    "warnings",
    "do_not_use",
    "ask_doctor_or_pharmacist",
)

JsonTransport = Callable[[str, Mapping[str, str], float, int], dict[str, Any]]


OpenFDASearchOutcome = InteractionSearchOutcome


class OpenFDASourceError(RuntimeError):
    """Safe source failure that never includes credentials or response bodies."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class OpenFDALabelClient:
    """Search official FDA labels for explicit mentions of an ingredient pair."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        label_url: str = OPENFDA_LABEL_URL,
        relay_secret: str | None = None,
        timeout: float = 15.0,
        max_results: int = 20,
        max_response_bytes: int = 8_000_000,
        transport: JsonTransport | None = None,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not 1 <= max_results <= 1000:
            raise ValueError("max_results must be between 1 and 1000")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if label_url not in APPROVED_OPENFDA_LABEL_URLS:
            raise ValueError("label_url is not an approved endpoint")
        if label_url == OPENFDA_RELAY_LABEL_URL:
            if not relay_secret:
                raise ValueError("relay endpoint requires relay_secret")
            if api_key:
                raise ValueError("relay endpoint does not accept api_key")
        elif relay_secret is not None:
            raise ValueError("relay_secret is only valid for the relay endpoint")
        self._api_key = api_key
        self._label_url = label_url
        self._relay_secret = relay_secret
        self._timeout = timeout
        self._max_results = max_results
        self._max_response_bytes = max_response_bytes
        self._transport = transport or _fetch_json

    def search_pair(
        self,
        ingredient_a: str,
        ingredient_b: str,
        *,
        checked_at: datetime | None = None,
    ) -> OpenFDASearchOutcome:
        first = _validate_ingredient(ingredient_a)
        second = _validate_ingredient(ingredient_b)
        if _normalize_text(first) == _normalize_text(second):
            raise ValueError("ingredients must be different")

        checked = checked_at or datetime.now(UTC)
        query_pair = f"{first} + {second}"
        failed_queries: list[str] = []
        missing_labels: list[str] = []
        documents: list[EvidenceDocument] = []
        latest_date: date | None = None

        for ingredient in (first, second):
            try:
                if not self._has_human_label(ingredient):
                    missing_labels.append(ingredient)
            except OpenFDASourceError:
                failed_queries.append(f"label_lookup:{ingredient}")

        if not failed_queries and not missing_labels:
            for subject, counterpart in ((first, second), (second, first)):
                try:
                    records, total = self._find_mentions(subject, counterpart)
                except OpenFDASourceError:
                    failed_queries.append(f"interaction_search:{subject}")
                    continue
                extracted, record_date = _records_to_documents(
                    records,
                    subject=subject,
                    counterpart=counterpart,
                    accessed_at=checked,
                )
                documents.extend(extracted)
                latest_date = _latest_date(latest_date, record_date)
                if total > len(records) and not extracted:
                    failed_queries.append(f"truncated_search:{subject}")

        complete = not failed_queries and not missing_labels
        if documents:
            status = SearchResultStatus.INTERACTION_FOUND
            details = (
                "Найдено релевантных фрагментов FDA labels: "
                f"{len(documents)}."
            )
            if not complete:
                details += " Часть запросов завершилась неполно."
        elif failed_queries:
            status = SearchResultStatus.FAILED
            details = "Обязательные запросы openFDA не завершены."
        elif missing_labels:
            status = SearchResultStatus.INCONCLUSIVE
            details = (
                "Не найдены подходящие human drug labels для: "
                + ", ".join(missing_labels)
            )
        else:
            status = SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
            details = (
                "В проверенных разделах openFDA Drug Label API прямое "
                "упоминание "
                "пары действующих веществ не найдено."
            )

        search_url = _public_search_url(first, second)
        search_result = SearchResult(
            source_id=_search_source_id(first, second),
            title=f"openFDA Drug Label search: {query_pair}",
            publisher=PUBLISHER,
            url=search_url,
            document_date=latest_date,
            accessed_at=checked,
            query_pairs=[query_pair],
            searched_sections=list(RELEVANT_SECTIONS),
            result=status,
            details=details,
        )
        return OpenFDASearchOutcome(
            provider_id="openfda.label",
            ingredient_pair=(first, second),
            search_result=search_result,
            documents=documents,
            complete=complete,
            failed_queries=failed_queries,
        )

    def _has_human_label(self, ingredient: str) -> bool:
        response = self._request(_identity_query(ingredient), limit=1)
        return bool(response.get("results"))

    def _find_mentions(
        self,
        subject: str,
        counterpart: str,
    ) -> tuple[list[dict[str, Any]], int]:
        response = self._request(
            _interaction_query(subject, counterpart),
            limit=self._max_results,
        )
        results = response.get("results", [])
        if not isinstance(results, list):
            raise OpenFDASourceError("invalid_results")
        records = [item for item in results if isinstance(item, dict)]
        total = response.get("meta", {}).get("results", {}).get("total", len(records))
        return records, total if isinstance(total, int) else len(records)

    def _request(self, search: str, *, limit: int) -> dict[str, Any]:
        params = {"search": search, "limit": str(limit)}
        if self._api_key:
            params["api_key"] = self._api_key
        url = f"{self._label_url}?{urlencode(params)}"
        headers = {"Accept": "application/json", "User-Agent": "PharmaCheckAI/0.1"}
        if self._relay_secret is not None:
            headers[OPENFDA_RELAY_SECRET_HEADER] = self._relay_secret
        try:
            return self._transport(
                url,
                headers,
                self._timeout,
                self._max_response_bytes,
            )
        except HTTPError as exc:
            if exc.code == 404:
                return {"meta": {"results": {"total": 0}}, "results": []}
            if exc.code == 429:
                raise OpenFDASourceError("rate_limited") from exc
            if 500 <= exc.code <= 599:
                raise OpenFDASourceError("upstream_unavailable") from exc
            raise OpenFDASourceError(f"http_{exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise OpenFDASourceError("network_error") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise OpenFDASourceError("invalid_json") from exc


def _fetch_json(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    max_response_bytes: int,
) -> dict[str, Any]:
    request = Request(url, headers=dict(headers), method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(max_response_bytes + 1)
    if len(payload) > max_response_bytes:
        raise OpenFDASourceError("response_too_large")
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise OpenFDASourceError("invalid_json_root")
    return decoded


def _identity_query(ingredient: str) -> str:
    identity = _identity_clause(ingredient)
    return f"{identity} AND {_human_product_clause()}"


def _interaction_query(subject: str, counterpart: str) -> str:
    sections = " OR ".join(
        f'{field}:"{_escape_query(counterpart)}"' for field in RELEVANT_SECTIONS
    )
    return (
        f"{_identity_clause(subject)} AND {_human_product_clause()}"
        f" AND ({sections})"
    )


def _identity_clause(ingredient: str) -> str:
    # The label can store a salt rather than the base ingredient (for example,
    # AMIODARONE HYDROCHLORIDE). The returned record is independently checked
    # by `_record_matches_subject`, so this query can safely match the term.
    escaped = _escape_query(ingredient.upper())
    return (
        f'(openfda.substance_name:"{escaped}" OR '
        f'openfda.generic_name:"{escaped}")'
    )


def _human_product_clause() -> str:
    values = " OR ".join(
        f'openfda.product_type.exact:"{value}"' for value in HUMAN_PRODUCT_TYPES
    )
    return f"({values})"


def _escape_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _validate_ingredient(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or len(normalized) > 200:
        raise ValueError("ingredient must contain 1 to 200 characters")
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        raise ValueError("ingredient contains control characters")
    return " ".join(normalized.split())


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", value).split())


def _contains_phrase(text: str, phrase: str) -> bool:
    haystack = f" {_normalize_text(text)} "
    needle = f" {_normalize_text(phrase)} "
    return needle in haystack


def _records_to_documents(
    records: list[dict[str, Any]],
    *,
    subject: str,
    counterpart: str,
    accessed_at: datetime,
) -> tuple[list[EvidenceDocument], date | None]:
    documents: list[EvidenceDocument] = []
    latest: date | None = None
    for record in records:
        if not _record_matches_subject(record, subject):
            continue
        record_date = _parse_effective_date(record.get("effective_time"))
        latest = _latest_date(latest, record_date)
        for section in RELEVANT_SECTIONS:
            values = record.get(section)
            if not isinstance(values, list):
                continue
            text_values = [value for value in values if isinstance(value, str)]
            excerpt = _extract_excerpt(text_values, counterpart)
            if not excerpt:
                continue
            document = _make_document(
                record,
                section=section,
                excerpt=excerpt,
                subject=subject,
                counterpart=counterpart,
                accessed_at=accessed_at,
                document_date=record_date,
            )
            documents.append(document)
    return documents, latest


def _record_matches_subject(record: dict[str, Any], subject: str) -> bool:
    openfda = record.get("openfda")
    if not isinstance(openfda, dict):
        return False
    expected = _normalize_text(subject)
    for field in ("substance_name", "generic_name"):
        values = openfda.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, str) and _contains_phrase(value, expected):
                return True
    return False


def _extract_excerpt(values: list[str], counterpart: str) -> str | None:
    snippets: list[str] = []
    for value in values:
        if not _contains_phrase(value, counterpart):
            continue
        collapsed = " ".join(value.split())
        if len(collapsed) <= 4000:
            snippets.append(collapsed)
            continue
        normalized_counterpart = _normalize_text(counterpart)
        match = re.search(re.escape(normalized_counterpart), _normalize_text(collapsed))
        if match is None:
            continue
        center = min(len(collapsed), match.start())
        start = max(0, center - 1500)
        end = min(len(collapsed), center + 2500)
        snippets.append(collapsed[start:end])
    if not snippets:
        return None
    return " […] ".join(snippets[:3])[:8000]


def _make_document(
    record: dict[str, Any],
    *,
    section: str,
    excerpt: str,
    subject: str,
    counterpart: str,
    accessed_at: datetime,
    document_date: date | None,
) -> EvidenceDocument:
    openfda = record.get("openfda") if isinstance(record.get("openfda"), dict) else {}
    set_id = _first_string(openfda.get("spl_set_id")) or str(record.get("set_id") or "")
    record_id = str(record.get("id") or set_id or "unknown")
    brand = _first_string(openfda.get("brand_name"))
    generic = _first_string(openfda.get("generic_name"))
    label_name = brand or generic or subject
    public_url = _record_url(set_id=set_id, record_id=record_id)
    digest = hashlib.sha256(
        f"{record_id}|{section}|{subject}|{counterpart}".encode()
    ).hexdigest()[:16]
    return EvidenceDocument(
        source_id=f"openfda.label.{digest}",
        source_tier=2,
        source_type="regulator_label",
        title=f"FDA label: {label_name}",
        publisher=PUBLISHER,
        url=public_url,
        document_date=document_date,
        accessed_at=accessed_at,
        language="en",
        applies_to=[subject, counterpart],
        section=section,
        spl_set_id=set_id or None,
        excerpt=excerpt,
        retrieval_status=RetrievalStatus.OK,
        matched_term=counterpart,
        matched_ingredient=counterpart,
    )


def _first_string(value: Any) -> str | None:
    if isinstance(value, list):
        return next((item for item in value if isinstance(item, str) and item), None)
    return value if isinstance(value, str) and value else None


def _record_url(*, set_id: str, record_id: str) -> str:
    if set_id:
        search = f'openfda.spl_set_id:"{_escape_query(set_id)}"'
    else:
        search = f'id:"{_escape_query(record_id)}"'
    return f"{OPENFDA_LABEL_URL}?{urlencode({'search': search, 'limit': '1'})}"


def _public_search_url(first: str, second: str) -> str:
    params = {
        "search": _interaction_query(first, second),
        "limit": "100",
    }
    return f"{OPENFDA_LABEL_URL}?{urlencode(params)}"


def _search_source_id(first: str, second: str) -> str:
    digest = hashlib.sha256(f"{first}|{second}".casefold().encode()).hexdigest()[:16]
    return f"openfda.label.search.{digest}"


def _parse_effective_date(value: Any) -> date | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        return None


def _latest_date(first: date | None, second: date | None) -> date | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)
