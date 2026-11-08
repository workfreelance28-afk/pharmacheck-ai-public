from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse

import pytest

from pharmacheck.contracts import RetrievalStatus, SearchResultStatus
from pharmacheck.sources.openfda import (
    OPENFDA_LABEL_URL,
    OPENFDA_RELAY_LABEL_URL,
    OPENFDA_RELAY_SECRET_HEADER,
    OpenFDALabelClient,
    _identity_query,
    _interaction_query,
)

CHECKED_AT = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


def _search_from_url(url: str) -> str:
    return parse_qs(urlparse(url).query)["search"][0]


def _label_record(*, ingredient: str, counterpart: str) -> dict[str, Any]:
    return {
        "id": "label-record-1",
        "effective_time": "20260115",
        "openfda": {
            "brand_name": ["Synthetic test label"],
            "generic_name": [ingredient],
            "substance_name": [ingredient.upper()],
            "spl_set_id": ["12345678-1234-1234-1234-123456789abc"],
        },
        "drug_interactions": [
            f"Synthetic fixture: concomitant use with {counterpart} may alter exposure."
        ],
    }


def test_interaction_found_creates_citable_document() -> None:
    calls: list[str] = []

    def transport(url: str, *_: object) -> dict[str, Any]:
        search = _search_from_url(url)
        calls.append(search)
        if "drug_interactions" not in search:
            return {"results": [{"id": "identity"}]}
        if "warfarin" in search.casefold() and "fluconazole" in search.casefold():
            return {
                "meta": {"results": {"total": 1}},
                "results": [
                    _label_record(ingredient="warfarin", counterpart="fluconazole")
                ],
            }
        return {"meta": {"results": {"total": 0}}, "results": []}

    outcome = OpenFDALabelClient(transport=transport).search_pair(
        "warfarin", "fluconazole", checked_at=CHECKED_AT
    )

    assert len(calls) == 4
    assert outcome.complete is True
    assert outcome.failed_queries == []
    assert outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    assert len(outcome.documents) == 1
    document = outcome.documents[0]
    assert document.source_tier == 2
    assert document.source_type == "regulator_label"
    assert document.section == "drug_interactions"
    assert document.retrieval_status is RetrievalStatus.OK
    assert document.document_date.isoformat() == "2026-01-15"
    assert "fluconazole" in document.excerpt
    assert "api_key" not in str(document.url)


def test_clean_two_direction_search_is_negative_not_safety_claim() -> None:
    mention_searches = 0

    def transport(url: str, *_: object) -> dict[str, Any]:
        nonlocal mention_searches
        search = _search_from_url(url)
        if "drug_interactions" not in search:
            return {"results": [{"id": "identity"}]}
        mention_searches += 1
        return {"meta": {"results": {"total": 0}}, "results": []}

    outcome = OpenFDALabelClient(transport=transport).search_pair(
        "ingredient one", "ingredient two", checked_at=CHECKED_AT
    )

    assert mention_searches == 2
    assert outcome.complete is True
    assert outcome.documents == []
    assert (
        outcome.search_result.result
        is SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
    )
    assert "безопас" not in (outcome.search_result.details or "").casefold()


def test_missing_human_label_is_inconclusive() -> None:
    calls = 0

    def transport(url: str, *_: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if '"known"' in _search_from_url(url):
            return {"results": [{"id": "identity"}]}
        return {"results": []}

    outcome = OpenFDALabelClient(transport=transport).search_pair(
        "known", "unknown", checked_at=CHECKED_AT
    )

    assert calls == 2
    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.INCONCLUSIVE
    assert outcome.documents == []


@pytest.mark.parametrize(
    ("exception", "expected_code"),
    [
        (HTTPError("https://api.fda.gov", 429, "rate", None, None), "label_lookup:a"),
        (HTTPError("https://api.fda.gov", 503, "down", None, None), "label_lookup:a"),
        (URLError("offline"), "label_lookup:a"),
        (TimeoutError(), "label_lookup:a"),
    ],
)
def test_source_failure_never_becomes_green_compatible_negative(
    exception: Exception,
    expected_code: str,
) -> None:
    def transport(url: str, *_: object) -> dict[str, Any]:
        if "a" in _search_from_url(url):
            raise exception
        return {"results": [{"id": "identity"}]}

    outcome = OpenFDALabelClient(transport=transport).search_pair(
        "a", "b", checked_at=CHECKED_AT
    )

    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.FAILED
    assert expected_code in outcome.failed_queries


def test_api_key_is_sent_but_never_exposed_in_returned_urls() -> None:
    request_urls: list[str] = []

    def transport(url: str, *_: object) -> dict[str, Any]:
        request_urls.append(url)
        search = _search_from_url(url)
        if "drug_interactions" not in search:
            return {"results": [{"id": "identity"}]}
        return {"meta": {"results": {"total": 0}}, "results": []}

    outcome = OpenFDALabelClient(
        api_key="secret-test-key", transport=transport
    ).search_pair("first", "second", checked_at=CHECKED_AT)

    assert request_urls and all("secret-test-key" in url for url in request_urls)
    assert "secret-test-key" not in str(outcome.search_result.url)
    assert "api_key" not in str(outcome.search_result.url)


def test_identity_terms_are_uppercased_and_allow_salt_names() -> None:
    query = _identity_query("ibuprofen")

    assert 'openfda.substance_name:"IBUPROFEN"' in query
    assert 'openfda.generic_name:"IBUPROFEN"' in query
    assert 'substance_name.exact:"IBUPROFEN"' not in query
    assert "+AND+" not in query
    assert "+OR+" not in query


def test_interaction_query_uses_logical_spaces_before_url_encoding() -> None:
    query = _interaction_query("ibuprofen", "warfarin")

    assert "+AND+" not in query
    assert "+OR+" not in query
    assert " AND " in query
    assert " OR " in query


def test_unrelated_openfda_record_is_not_accepted_as_subject_evidence() -> None:
    def transport(url: str, *_: object) -> dict[str, Any]:
        search = _search_from_url(url)
        if "drug_interactions" not in search:
            return {"results": [{"id": "identity"}]}
        return {
            "meta": {"results": {"total": 1}},
            "results": [
                {
                    "id": "unrelated",
                    "openfda": {
                        "generic_name": ["CYCLOSPORINE"],
                        "substance_name": ["CYCLOSPORINE"],
                    },
                    "drug_interactions": [
                        "An unrelated list mentions amiodarone and voriconazole."
                    ],
                }
            ],
        }

    outcome = OpenFDALabelClient(transport=transport).search_pair(
        "amiodarone", "voriconazole", checked_at=CHECKED_AT
    )

    assert outcome.complete is True
    assert outcome.documents == []
    assert (
        outcome.search_result.result
        is SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
    )


def test_relay_endpoint_uses_secret_header_but_citations_remain_official() -> None:
    requests: list[tuple[str, dict[str, str]]] = []

    def transport(url: str, headers: object, *_: object) -> dict[str, Any]:
        assert isinstance(headers, dict)
        requests.append((url, headers))
        if "drug_interactions" not in _search_from_url(url):
            return {"results": [{"id": "identity"}]}
        return {"meta": {"results": {"total": 0}}, "results": []}

    outcome = OpenFDALabelClient(
        label_url=OPENFDA_RELAY_LABEL_URL,
        relay_secret="test-relay-secret",
        transport=transport,
    ).search_pair("ibuprofen", "warfarin", checked_at=CHECKED_AT)

    assert requests
    assert all(url.startswith(OPENFDA_RELAY_LABEL_URL) for url, _ in requests)
    assert all(
        headers[OPENFDA_RELAY_SECRET_HEADER] == "test-relay-secret"
        for _, headers in requests
    )
    public_url = str(outcome.search_result.url)
    assert public_url.startswith(OPENFDA_LABEL_URL)
    assert "test-relay-secret" not in public_url


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label_url": "https://untrusted.example.test/drug/label.json"},
        {"label_url": OPENFDA_RELAY_LABEL_URL},
        {"relay_secret": "unexpected"},
        {
            "label_url": OPENFDA_RELAY_LABEL_URL,
            "relay_secret": "secret",
            "api_key": "unsupported-through-relay",
        },
    ],
)
def test_unapproved_or_inconsistent_relay_configuration_is_rejected(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError):
        OpenFDALabelClient(**kwargs)


@pytest.mark.parametrize("value", ["", "   ", "bad\nvalue", "x" * 201])
def test_invalid_ingredient_is_rejected_before_network(value: str) -> None:
    client = OpenFDALabelClient(transport=lambda *_: pytest.fail("network called"))

    with pytest.raises(ValueError):
        client.search_pair(value, "valid")


def test_same_ingredients_are_rejected() -> None:
    client = OpenFDALabelClient(transport=lambda *_: pytest.fail("network called"))

    with pytest.raises(ValueError, match="different"):
        client.search_pair("Warfarin", " warfarin ")
