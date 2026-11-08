"""Targeted adapter for official DailyMed Structured Product Labels."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
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
from pharmacheck.sources.rxclass import (
    DrugClassMembership,
    RxClassClient,
    RxClassSourceError,
    class_lookup_url,
)
from pharmacheck.sources.rxnorm import (
    RxNormClient,
    RxNormSourceError,
    RxNormStatus,
)

DAILYMED_SPLS_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls"
DAILYMED_PUBLIC_URL = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"
PUBLISHER = "U.S. National Library of Medicine (DailyMed)"
RELEVANT_SECTION_CODES = {
    "34066-1": "boxed_warning",
    "34073-7": "drug_interactions",
    "34070-3": "contraindications",
    "34071-1": "warnings",
    "42232-9": "precautions",
    "43685-7": "warnings_and_precautions",
}

BytesTransport = Callable[[str, Mapping[str, str], float, int], bytes]
DailyMedSearchOutcome = InteractionSearchOutcome


class DailyMedSourceError(RuntimeError):
    """Safe source failure that never includes response bodies."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class DailyMedLabelClient:
    """Find explicit pair mentions in a small number of current SPL labels."""

    def __init__(
        self,
        *,
        rxnorm: RxNormClient,
        timeout: float = 12.0,
        max_labels_per_ingredient: int = 2,
        max_response_bytes: int = 4_000_000,
        transport: BytesTransport | None = None,
        rxclass: RxClassClient | None = None,
    ) -> None:
        if timeout <= 0 or max_response_bytes <= 0:
            raise ValueError("DailyMed client limits must be positive")
        if not 1 <= max_labels_per_ingredient <= 3:
            raise ValueError("max_labels_per_ingredient must be between 1 and 3")
        self._rxnorm = rxnorm
        self._timeout = timeout
        self._max_labels = max_labels_per_ingredient
        self._max_response_bytes = max_response_bytes
        self._transport = transport or _fetch_bytes
        self._rxclass = rxclass

    def search_pair(
        self,
        ingredient_a: str,
        ingredient_b: str,
        *,
        checked_at: datetime | None = None,
    ) -> DailyMedSearchOutcome:
        first = _validate_ingredient(ingredient_a)
        second = _validate_ingredient(ingredient_b)
        if _normalize_text(first) == _normalize_text(second):
            raise ValueError("ingredients must be different")

        checked = checked_at or datetime.now(UTC)
        failed_queries: list[str] = []
        missing_labels: list[str] = []
        documents: list[EvidenceDocument] = []
        latest_date: date | None = None
        search_urls: list[str] = []
        rxcuis: dict[str, str | None] = {}
        class_memberships: dict[str, tuple[DrugClassMembership, ...]] = {}

        for ingredient in (first, second):
            try:
                rxcuis[ingredient] = self._resolve_rxcui(ingredient)
            except RxNormSourceError:
                failed_queries.append(f"label_lookup:{ingredient}")

        if self._rxclass is not None:
            for ingredient in (first, second):
                rxcui = rxcuis.get(ingredient)
                if rxcui is None:
                    failed_queries.append(f"class_lookup:{ingredient}")
                    continue
                try:
                    class_memberships[ingredient] = self._rxclass.get_classes(rxcui)
                except RxClassSourceError:
                    failed_queries.append(f"class_lookup:{ingredient}")

        for subject, counterpart in ((first, second), (second, first)):
            if subject not in rxcuis:
                continue
            try:
                labels, search_url = self._find_labels(subject, rxcuis[subject])
                search_urls.append(search_url)
            except DailyMedSourceError:
                failed_queries.append(f"label_lookup:{subject}")
                continue

            if not labels:
                missing_labels.append(subject)
                continue

            relevant_sections_found = False
            for label in labels[: self._max_labels]:
                setid = label.get("setid")
                if not isinstance(setid, str) or not _is_setid(setid):
                    failed_queries.append(f"invalid_setid:{subject}")
                    continue
                label_date = _parse_published_date(label.get("published_date"))
                latest_date = _latest_date(latest_date, label_date)
                try:
                    xml_payload = self._request(f"{DAILYMED_SPLS_URL}/{setid}.xml")
                    extracted, has_relevant_sections = _xml_to_documents(
                        xml_payload,
                        setid=setid,
                        label_title=_label_title(label, subject),
                        subject=subject,
                        counterpart=counterpart,
                        counterpart_classes=class_memberships.get(counterpart, ()),
                        accessed_at=checked,
                        document_date=label_date,
                    )
                except DailyMedSourceError:
                    failed_queries.append(f"spl_fetch:{subject}:{setid}")
                    continue
                relevant_sections_found = (
                    relevant_sections_found or has_relevant_sections
                )
                documents.extend(extracted)
            if not relevant_sections_found:
                failed_queries.append(f"relevant_sections_missing:{subject}")

        complete = not failed_queries and not missing_labels
        if documents:
            status = SearchResultStatus.INTERACTION_FOUND
            details = f"Найдено релевантных фрагментов DailyMed: {len(documents)}."
            if not complete:
                details += " Часть запросов завершилась неполно."
        elif failed_queries:
            status = SearchResultStatus.FAILED
            details = "Обязательные запросы DailyMed не завершены."
        elif missing_labels:
            status = SearchResultStatus.INCONCLUSIVE
            details = "Не найдены подходящие DailyMed labels для: " + ", ".join(
                missing_labels
            )
        else:
            status = SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
            details = (
                "В проверенных разделах выбранных DailyMed labels прямое упоминание "
                "пары действующих веществ не найдено."
            )

        search_result = SearchResult(
            source_id=_search_source_id(first, second),
            title=f"DailyMed label search: {first} + {second}",
            publisher=PUBLISHER,
            url=search_urls[0] if search_urls else _name_search_url(first),
            document_date=latest_date,
            accessed_at=checked,
            query_pairs=[f"{first} + {second}"],
            searched_sections=list(dict.fromkeys(RELEVANT_SECTION_CODES.values())),
            additional_urls=search_urls[1:],
            result=status,
            details=details,
        )
        return DailyMedSearchOutcome(
            provider_id="dailymed.label",
            ingredient_pair=(first, second),
            search_result=search_result,
            documents=documents,
            complete=complete,
            failed_queries=failed_queries,
        )

    def _resolve_rxcui(self, ingredient: str) -> str | None:
        resolution = self._rxnorm.resolve(ingredient)
        if resolution.status is RxNormStatus.MATCHED:
            return resolution.rxcui
        return None

    def _find_labels(
        self,
        ingredient: str,
        rxcui: str | None,
    ) -> tuple[list[dict[str, Any]], str]:
        params = {
            "pagesize": str(self._max_labels),
            "page": "1",
        }
        if rxcui:
            params["rxcui"] = rxcui
        else:
            params.update({"drug_name": ingredient, "name_type": "generic"})
        url = f"{DAILYMED_SPLS_URL}.json?{urlencode(params)}"
        payload = self._request(url)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise DailyMedSourceError("invalid_json") from exc
        if not isinstance(decoded, dict):
            raise DailyMedSourceError("invalid_json_root")
        raw = decoded.get("data", [])
        if not isinstance(raw, list):
            raise DailyMedSourceError("invalid_results")
        labels = [item for item in raw if isinstance(item, dict)]
        labels.sort(
            key=lambda item: _parse_published_date(item.get("published_date"))
            or date.min,
            reverse=True,
        )
        return labels, url

    def _request(self, url: str) -> bytes:
        try:
            return self._transport(
                url,
                {
                    "Accept": "application/json, application/xml;q=0.9",
                    "User-Agent": "PharmaCheckAI/0.1",
                },
                self._timeout,
                self._max_response_bytes,
            )
        except HTTPError as exc:
            if exc.code == 429:
                raise DailyMedSourceError("rate_limited") from exc
            if 500 <= exc.code <= 599:
                raise DailyMedSourceError("upstream_unavailable") from exc
            raise DailyMedSourceError(f"http_{exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise DailyMedSourceError("network_error") from exc


def _fetch_bytes(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    max_response_bytes: int,
) -> bytes:
    request = Request(url, headers=dict(headers), method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(max_response_bytes + 1)
    if len(payload) > max_response_bytes:
        raise DailyMedSourceError("response_too_large")
    return payload


def _xml_to_documents(
    payload: bytes,
    *,
    setid: str,
    label_title: str,
    subject: str,
    counterpart: str,
    accessed_at: datetime,
    document_date: date | None,
    counterpart_classes: tuple[DrugClassMembership, ...] = (),
) -> tuple[list[EvidenceDocument], bool]:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise DailyMedSourceError("invalid_xml") from exc

    documents: list[EvidenceDocument] = []
    relevant_sections_found = False
    for section in root.iter():
        if _local_name(section.tag) != "section":
            continue
        code = _direct_child_attribute(section, "code", "code")
        section_name = RELEVANT_SECTION_CODES.get(code or "")
        if section_name is None:
            continue
        relevant_sections_found = True
        text = _section_text(section)
        if not text:
            continue
        excerpt = _extract_excerpt(text, counterpart)
        matched_class: DrugClassMembership | None = None
        if excerpt is None:
            for membership in sorted(
                counterpart_classes,
                key=lambda item: (-len(item.class_name), item.class_name.casefold()),
            ):
                excerpt = _extract_excerpt(
                    text,
                    membership.class_name,
                    allow_plural=True,
                )
                if excerpt is not None:
                    matched_class = membership
                    break
        if excerpt is None:
            continue
        digest = hashlib.sha256(
            (
                f"{setid}|{section_name}|{subject}|{counterpart}|"
                f"{matched_class.class_id if matched_class else 'direct'}"
            ).encode()
        ).hexdigest()[:16]
        documents.append(
            EvidenceDocument(
                source_id=f"dailymed.label.{digest}",
                source_tier=2,
                source_type="regulator_label",
                title=f"DailyMed label: {label_title}",
                publisher=PUBLISHER,
                url=f"{DAILYMED_PUBLIC_URL}?{urlencode({'setid': setid})}",
                document_date=document_date,
                accessed_at=accessed_at,
                language="en",
                applies_to=[subject, counterpart],
                section=section_name,
                spl_set_id=setid,
                excerpt=excerpt,
                retrieval_status=RetrievalStatus.OK,
                match_basis=(
                    "structured_class" if matched_class else "direct_ingredient"
                ),
                matched_term=matched_class.class_name if matched_class else counterpart,
                matched_ingredient=counterpart,
                additional_urls=(
                    [
                        class_lookup_url(
                            matched_class.rxcui,
                            matched_class.relation_source,
                        )
                    ]
                    if matched_class
                    else []
                ),
            )
        )
    return documents, relevant_sections_found


def _section_text(section: ET.Element) -> str:
    """Collect text from direct and nested SPL subsections."""

    fragments = [
        " ".join(" ".join(element.itertext()).split())
        for element in section.iter()
        if _local_name(element.tag) == "text"
    ]
    return " ".join(fragment for fragment in fragments if fragment)


def _extract_excerpt(
    text: str,
    counterpart: str,
    *,
    allow_plural: bool = False,
) -> str | None:
    words = re.findall(r"[\w]+", unicodedata.normalize("NFKC", counterpart))
    if not words:
        return None
    escaped = [re.escape(word) for word in words]
    if allow_plural:
        final_word = words[-1]
        # RxClass can expose a class as a plural noun (for example "Azoles"),
        # while an SPL label uses the singular form as a modifier ("azole
        # antifungals").  Accept the exact plural and its singular form without
        # broad substring matching.
        if len(final_word) > 3 and final_word.casefold().endswith("s"):
            escaped[-1] = re.escape(final_word[:-1]) + "s?"
        else:
            escaped[-1] += r"(?:s|es)?"
    pattern = re.compile(r"\b" + r"[\W_]+".join(escaped) + r"\b", re.I)
    match = pattern.search(text)
    if match is None:
        return None
    start = max(0, match.start() - 1_200)
    end = min(len(text), match.end() + 2_800)
    prefix = "…" if start else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"[:4_100]


def _direct_child(element: ET.Element, local_name: str) -> ET.Element | None:
    return next(
        (child for child in element if _local_name(child.tag) == local_name),
        None,
    )


def _direct_child_attribute(
    element: ET.Element,
    child_name: str,
    attribute: str,
) -> str | None:
    child = _direct_child(element, child_name)
    return child.get(attribute) if child is not None else None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _label_title(label: dict[str, Any], fallback: str) -> str:
    value = label.get("title")
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _parse_published_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    for date_format in ("%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).date()
        except ValueError:
            continue
    return None


def _latest_date(first: date | None, second: date | None) -> date | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def _is_setid(value: str) -> bool:
    return re.fullmatch(r"[0-9a-fA-F-]{32,36}", value) is not None


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


def _name_search_url(ingredient: str) -> str:
    params = {
        "drug_name": ingredient,
        "name_type": "generic",
        "pagesize": "1",
        "page": "1",
    }
    return f"{DAILYMED_SPLS_URL}.json?{urlencode(params)}"


def _search_source_id(first: str, second: str) -> str:
    digest = hashlib.sha256(f"{first}|{second}".casefold().encode()).hexdigest()[:16]
    return f"dailymed.label.search.{digest}"
