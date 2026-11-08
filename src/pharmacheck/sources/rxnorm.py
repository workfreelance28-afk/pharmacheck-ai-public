"""Small, cached client for the public NLM RxNorm API."""

from __future__ import annotations

import json
import time
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RXNORM_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
RXNORM_PROPERTIES_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
RXNORM_RELATED_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN"

JsonTransport = Callable[[str, Mapping[str, str], float, int], dict[str, Any]]


class RxNormStatus(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True, slots=True)
class RxNormResolution:
    input_name: str
    status: RxNormStatus
    rxcui: str | None = None


@dataclass(frozen=True, slots=True)
class RxNormConcept:
    rxcui: str
    name: str
    tty: str
    language: str
    suppress: str


class RxNormSourceError(RuntimeError):
    """Safe upstream failure that does not expose response data."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class RxNormClient:
    """Resolve an ingredient name to one active RxNorm concept identifier."""

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
            raise ValueError("RxNorm client limits must be valid")
        self._timeout = timeout
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_response_bytes = max_response_bytes
        self._transport = transport or _fetch_json
        self._clock = clock
        self._cache: dict[str, tuple[float, RxNormResolution]] = {}

    def resolve(self, ingredient: str) -> RxNormResolution:
        name = _validate_ingredient(ingredient)
        cache_key = _normalize_name(name)
        cached = self._cache.get(cache_key)
        now = self._clock()
        if cached is not None and cached[0] >= now:
            return cached[1]

        params = {
            "name": name,
            "search": "2",
            "allsrc": "0",
        }
        url = f"{RXNORM_RXCUI_URL}?{urlencode(params)}"
        try:
            response = self._transport(
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
                raise RxNormSourceError("rate_limited") from exc
            if 500 <= exc.code <= 599:
                raise RxNormSourceError("upstream_unavailable") from exc
            raise RxNormSourceError(f"http_{exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise RxNormSourceError("network_error") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RxNormSourceError("invalid_json") from exc

        identifiers = _extract_identifiers(response)
        if len(identifiers) == 1:
            resolution = RxNormResolution(
                input_name=name,
                status=RxNormStatus.MATCHED,
                rxcui=identifiers[0],
            )
        elif identifiers:
            resolution = RxNormResolution(
                input_name=name,
                status=RxNormStatus.AMBIGUOUS,
            )
        else:
            resolution = RxNormResolution(
                input_name=name,
                status=RxNormStatus.NOT_FOUND,
            )
        self._cache[cache_key] = (now + self._cache_ttl_seconds, resolution)
        return resolution

    def get_concept(self, rxcui: str) -> RxNormConcept:
        if not rxcui.isdigit() or len(rxcui) > 20:
            raise ValueError("rxcui must contain 1 to 20 digits")
        url = RXNORM_PROPERTIES_URL.format(rxcui=rxcui)
        try:
            response = self._transport(
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
                raise RxNormSourceError("rate_limited") from exc
            if 500 <= exc.code <= 599:
                raise RxNormSourceError("upstream_unavailable") from exc
            raise RxNormSourceError(f"http_{exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise RxNormSourceError("network_error") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RxNormSourceError("invalid_json") from exc
        return _extract_concept(response, expected_rxcui=rxcui)

    def get_related_ingredient(self, rxcui: str) -> RxNormConcept | None:
        """Return the single RxNorm `IN` concept related to a `PIN`, if any.

        A `PIN` (Precise Ingredient, e.g. a specific salt form) always names
        an exact substance, but production evidence sources (DDInter) match
        ingredients by name string against their own `IN`-level vocabulary.
        Resolving to the parent `IN` keeps naming consistent instead of
        silently loosening the ingredient-identity check to accept `PIN`.
        """
        if not rxcui.isdigit() or len(rxcui) > 20:
            raise ValueError("rxcui must contain 1 to 20 digits")
        url = RXNORM_RELATED_URL.format(rxcui=rxcui)
        try:
            response = self._transport(
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
                raise RxNormSourceError("rate_limited") from exc
            if 500 <= exc.code <= 599:
                raise RxNormSourceError("upstream_unavailable") from exc
            raise RxNormSourceError(f"http_{exc.code}") from exc
        except (TimeoutError, URLError) as exc:
            raise RxNormSourceError("network_error") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RxNormSourceError("invalid_json") from exc
        return _extract_single_related_in(response)


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
        raise RxNormSourceError("response_too_large")
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RxNormSourceError("invalid_json_root")
    return decoded


def _extract_identifiers(response: dict[str, Any]) -> list[str]:
    group = response.get("idGroup")
    if not isinstance(group, dict):
        raise RxNormSourceError("invalid_response")
    raw = group.get("rxnormId", [])
    if raw is None:
        return []
    if not isinstance(raw, list) or any(
        not isinstance(value, str) or not value.isdigit() for value in raw
    ):
        raise RxNormSourceError("invalid_response")
    return list(dict.fromkeys(raw))


def _extract_concept(response: dict[str, Any], *, expected_rxcui: str) -> RxNormConcept:
    properties = response.get("properties")
    if not isinstance(properties, dict):
        raise RxNormSourceError("invalid_response")
    required = ("rxcui", "name", "tty", "language", "suppress")
    if any(not isinstance(properties.get(key), str) for key in required):
        raise RxNormSourceError("invalid_response")
    if properties["rxcui"] != expected_rxcui or not properties["name"].strip():
        raise RxNormSourceError("invalid_response")
    return RxNormConcept(
        rxcui=properties["rxcui"],
        name=properties["name"],
        tty=properties["tty"],
        language=properties["language"],
        suppress=properties["suppress"],
    )


def _extract_single_related_in(response: dict[str, Any]) -> RxNormConcept | None:
    group = response.get("relatedGroup")
    if not isinstance(group, dict):
        raise RxNormSourceError("invalid_response")
    concept_groups = group.get("conceptGroup")
    if concept_groups is None:
        return None
    if not isinstance(concept_groups, list):
        raise RxNormSourceError("invalid_response")
    concepts: list[RxNormConcept] = []
    for concept_group in concept_groups:
        if not isinstance(concept_group, dict) or concept_group.get("tty") != "IN":
            continue
        properties = concept_group.get("conceptProperties") or []
        if not isinstance(properties, list):
            raise RxNormSourceError("invalid_response")
        required = ("rxcui", "name", "tty", "language", "suppress")
        for item in properties:
            if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) for key in required
            ):
                raise RxNormSourceError("invalid_response")
            concepts.append(
                RxNormConcept(
                    rxcui=item["rxcui"],
                    name=item["name"],
                    tty=item["tty"],
                    language=item["language"],
                    suppress=item["suppress"],
                )
            )
    if len(concepts) != 1:
        return None
    return concepts[0]


def _validate_ingredient(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or len(normalized) > 200:
        raise ValueError("ingredient must contain 1 to 200 characters")
    if any(unicodedata.category(char).startswith("C") for char in normalized):
        raise ValueError("ingredient contains control characters")
    return " ".join(normalized.split())


def _normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
