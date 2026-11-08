"""Small cached client for official FDA SPL drug classes exposed by RxClass."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RXCLASS_BY_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"
RXCLASS_RELATION_SOURCES = ("DailyMed", "FDASPL")
ALLOWED_CLASS_TYPES = frozenset({"EPC", "CHEM", "MOA", "PE"})
ALLOWED_RELATIONS = frozenset(
    {"has_epc", "has_chemical_structure", "has_moa", "has_pe"}
)

JsonTransport = Callable[[str, Mapping[str, str], float, int], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class DrugClassMembership:
    rxcui: str
    class_id: str
    class_name: str
    class_type: str
    relation: str
    relation_source: str


class RxClassSourceError(RuntimeError):
    """Safe upstream failure that never exposes response data."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RxClassClient:
    """Return direct FDA SPL class memberships for an RxNorm ingredient."""

    def __init__(
        self,
        *,
        timeout: float = 8.0,
        cache_ttl_seconds: float = 86_400,
        max_response_bytes: int = 500_000,
        transport: JsonTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout <= 0 or cache_ttl_seconds < 0 or max_response_bytes <= 0:
            raise ValueError("RxClass client limits must be valid")
        self._timeout = timeout
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_response_bytes = max_response_bytes
        self._transport = transport or _fetch_json
        self._clock = clock
        self._cache: dict[str, tuple[float, tuple[DrugClassMembership, ...]]] = {}

    def get_classes(self, rxcui: str) -> tuple[DrugClassMembership, ...]:
        _validate_rxcui(rxcui)
        cached = self._cache.get(rxcui)
        now = self._clock()
        if cached is not None and cached[0] >= now:
            return cached[1]

        last_error: RxClassSourceError | None = None
        memberships: tuple[DrugClassMembership, ...] = ()
        for relation_source in RXCLASS_RELATION_SOURCES:
            try:
                response = self._request(class_lookup_url(rxcui, relation_source))
                memberships = _extract_memberships(
                    response,
                    expected_rxcui=rxcui,
                    expected_source=relation_source,
                )
            except RxClassSourceError as exc:
                last_error = exc
                continue
            if memberships:
                break
        if not memberships and last_error is not None:
            raise last_error
        self._cache[rxcui] = (now + self._cache_ttl_seconds, memberships)
        return memberships

    def _request(self, url: str) -> dict[str, Any]:
        try:
            return self._transport(
                url,
                {
                    "Accept": "application/json",
                    "User-Agent": "PharmaCheckAI/0.1",
                },
                self._timeout,
                self._max_response_bytes,
            )
        except HTTPError as exc:
            if exc.code == 429:
                raise RxClassSourceError("rate_limited") from exc
            if 500 <= exc.code <= 599:
                raise RxClassSourceError("upstream_unavailable") from exc
            raise RxClassSourceError(f"http_{exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise RxClassSourceError("network_error") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RxClassSourceError("invalid_json") from exc


def class_lookup_url(rxcui: str, relation_source: str = "DailyMed") -> str:
    _validate_rxcui(rxcui)
    if relation_source not in RXCLASS_RELATION_SOURCES:
        raise ValueError("relation_source is not approved")
    params = {"rxcui": rxcui, "relaSource": relation_source}
    return f"{RXCLASS_BY_RXCUI_URL}?{urlencode(params)}"


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
        raise RxClassSourceError("response_too_large")
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RxClassSourceError("invalid_json_root")
    return decoded


def _extract_memberships(
    response: dict[str, Any], *, expected_rxcui: str, expected_source: str
) -> tuple[DrugClassMembership, ...]:
    # RxClass uses an empty JSON object as the documented wire response when
    # the requested relation source has no class memberships for an RxCUI.
    if not response:
        return ()
    if "rxclassDrugInfoList" not in response:
        raise RxClassSourceError("invalid_response")
    raw_list = response.get("rxclassDrugInfoList")
    if raw_list is None:
        return ()
    if not isinstance(raw_list, dict):
        raise RxClassSourceError("invalid_response")
    raw_items = raw_list.get("rxclassDrugInfo", [])
    if raw_items is None:
        return ()
    if not isinstance(raw_items, list):
        raise RxClassSourceError("invalid_response")

    memberships: dict[tuple[str, str], DrugClassMembership] = {}
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise RxClassSourceError("invalid_response")
        drug = raw.get("minConcept")
        drug_class = raw.get("rxclassMinConceptItem")
        relation = raw.get("rela")
        source = raw.get("relaSource")
        if not isinstance(drug, dict) or not isinstance(drug_class, dict):
            raise RxClassSourceError("invalid_response")
        if drug.get("rxcui") != expected_rxcui:
            continue
        class_id = drug_class.get("classId")
        class_name = drug_class.get("className")
        class_type = drug_class.get("classType")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (class_id, class_name, class_type, relation, source)
        ):
            raise RxClassSourceError("invalid_response")
        normalized_relation = relation.casefold()
        normalized_type = class_type.upper()
        if (
            source.casefold() != expected_source.casefold()
            or normalized_type not in ALLOWED_CLASS_TYPES
            or normalized_relation not in ALLOWED_RELATIONS
        ):
            continue
        membership = DrugClassMembership(
            rxcui=expected_rxcui,
            class_id=class_id,
            class_name=" ".join(class_name.split()),
            class_type=normalized_type,
            relation=normalized_relation,
            relation_source=expected_source,
        )
        memberships[(membership.class_id, membership.relation)] = membership
    return tuple(memberships[key] for key in sorted(memberships))


def _validate_rxcui(value: str) -> None:
    if not value.isdigit() or len(value) > 20:
        raise ValueError("rxcui must contain 1 to 20 digits")
