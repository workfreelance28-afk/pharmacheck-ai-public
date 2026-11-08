from __future__ import annotations

from urllib.error import HTTPError, URLError

import pytest

from pharmacheck.sources.rxnorm import (
    RxNormClient,
    RxNormSourceError,
    RxNormStatus,
)


def test_unique_normalized_match_is_returned_and_cached() -> None:
    calls = 0

    def transport(*_: object):
        nonlocal calls
        calls += 1
        return {"idGroup": {"rxnormId": ["5640"]}}

    client = RxNormClient(transport=transport)

    first = client.resolve(" Ibuprofen ")
    second = client.resolve("ibuprofen")

    assert first.status is RxNormStatus.MATCHED
    assert first.rxcui == "5640"
    assert second == first
    assert calls == 1


def test_no_match_and_ambiguous_match_are_not_guessed() -> None:
    responses = iter(
        [
            {"idGroup": {}},
            {"idGroup": {"rxnormId": ["1", "2"]}},
        ]
    )
    client = RxNormClient(transport=lambda *_: next(responses))

    assert client.resolve("unknown").status is RxNormStatus.NOT_FOUND
    ambiguous = client.resolve("ambiguous")
    assert ambiguous.status is RxNormStatus.AMBIGUOUS
    assert ambiguous.rxcui is None


@pytest.mark.parametrize(
    "exception",
    [
        HTTPError("https://rxnav.nlm.nih.gov", 429, "rate", None, None),
        HTTPError("https://rxnav.nlm.nih.gov", 503, "down", None, None),
        URLError("offline"),
        TimeoutError(),
    ],
)
def test_upstream_failures_are_safe(exception: Exception) -> None:
    client = RxNormClient(transport=lambda *_: (_ for _ in ()).throw(exception))

    with pytest.raises(RxNormSourceError):
        client.resolve("ibuprofen")


@pytest.mark.parametrize("value", ["", "  ", "bad\nvalue", "x" * 201])
def test_invalid_name_is_rejected_before_network(value: str) -> None:
    client = RxNormClient(transport=lambda *_: pytest.fail("network called"))

    with pytest.raises(ValueError):
        client.resolve(value)


def test_invalid_response_is_rejected() -> None:
    client = RxNormClient(transport=lambda *_: {"unexpected": {}})

    with pytest.raises(RxNormSourceError, match="invalid_response"):
        client.resolve("ibuprofen")


def test_concept_properties_are_validated() -> None:
    client = RxNormClient(
        transport=lambda *_: {
            "properties": {
                "rxcui": "321988",
                "name": "escitalopram",
                "tty": "IN",
                "language": "ENG",
                "suppress": "N",
            }
        }
    )

    concept = client.get_concept("321988")

    assert concept.name == "escitalopram"
    assert concept.tty == "IN"


def test_concept_rejects_wrong_identifier_and_invalid_input() -> None:
    client = RxNormClient(
        transport=lambda *_: {
            "properties": {
                "rxcui": "different",
                "name": "escitalopram",
                "tty": "IN",
                "language": "ENG",
                "suppress": "N",
            }
        }
    )

    with pytest.raises(RxNormSourceError, match="invalid_response"):
        client.get_concept("321988")
    with pytest.raises(ValueError):
        client.get_concept("not-a-number")


def test_related_ingredient_returns_the_single_matching_in_concept() -> None:
    client = RxNormClient(
        transport=lambda *_: {
            "relatedGroup": {
                "rxcui": None,
                "conceptGroup": [
                    {
                        "tty": "IN",
                        "conceptProperties": [
                            {
                                "rxcui": "7213",
                                "name": "ipratropium",
                                "synonym": "",
                                "tty": "IN",
                                "language": "ENG",
                                "suppress": "N",
                                "umlscui": "",
                            }
                        ],
                    }
                ],
            }
        }
    )

    parent = client.get_related_ingredient("203212")

    assert parent is not None
    assert parent.rxcui == "7213"
    assert parent.name == "ipratropium"
    assert parent.tty == "IN"


def test_related_ingredient_is_none_when_absent_or_ambiguous() -> None:
    empty_client = RxNormClient(
        transport=lambda *_: {"relatedGroup": {"rxcui": None, "conceptGroup": []}}
    )
    assert empty_client.get_related_ingredient("1") is None

    two_matches = {
        "relatedGroup": {
            "rxcui": None,
            "conceptGroup": [
                {
                    "tty": "IN",
                    "conceptProperties": [
                        {
                            "rxcui": "1",
                            "name": "a",
                            "synonym": "",
                            "tty": "IN",
                            "language": "ENG",
                            "suppress": "N",
                            "umlscui": "",
                        },
                        {
                            "rxcui": "2",
                            "name": "b",
                            "synonym": "",
                            "tty": "IN",
                            "language": "ENG",
                            "suppress": "N",
                            "umlscui": "",
                        },
                    ],
                }
            ],
        }
    }
    ambiguous_client = RxNormClient(transport=lambda *_: two_matches)
    assert ambiguous_client.get_related_ingredient("1") is None


def test_related_ingredient_rejects_invalid_input_and_response() -> None:
    client = RxNormClient(transport=lambda *_: {"unexpected": {}})

    with pytest.raises(RxNormSourceError, match="invalid_response"):
        client.get_related_ingredient("1")
    with pytest.raises(ValueError):
        client.get_related_ingredient("not-a-number")
