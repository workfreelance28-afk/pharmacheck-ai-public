from __future__ import annotations

from urllib.error import HTTPError, URLError

import pytest

from pharmacheck.sources.rxclass import (
    RxClassClient,
    RxClassSourceError,
    class_lookup_url,
)


def _response() -> dict:
    return {
        "rxclassDrugInfoList": {
            "rxclassDrugInfo": [
                {
                    "minConcept": {
                        "rxcui": "121243",
                        "name": "voriconazole",
                        "tty": "IN",
                    },
                    "rxclassMinConceptItem": {
                        "classId": "N0000174037",
                        "className": "Azole Antifungal",
                        "classType": "EPC",
                    },
                    "rela": "has_EPC",
                    "relaSource": "DailyMed",
                },
                {
                    "minConcept": {
                        "rxcui": "999",
                        "name": "unrelated",
                        "tty": "IN",
                    },
                    "rxclassMinConceptItem": {
                        "classId": "ignored",
                        "className": "Unrelated Class",
                        "classType": "EPC",
                    },
                    "rela": "has_EPC",
                    "relaSource": "DailyMed",
                },
            ]
        }
    }


def test_direct_fdaspl_memberships_are_returned_and_cached() -> None:
    calls = []

    def transport(url, *_):
        calls.append(url)
        return _response()

    client = RxClassClient(transport=transport)

    first = client.get_classes("121243")
    second = client.get_classes("121243")

    assert first == second
    assert len(first) == 1
    assert first[0].class_name == "Azole Antifungal"
    assert first[0].class_type == "EPC"
    assert calls == [class_lookup_url("121243")]


def test_fdaspl_is_used_as_fallback_when_dailymed_has_no_memberships() -> None:
    def transport(url, *_):
        if "relaSource=DailyMed" in url:
            return {"rxclassDrugInfoList": {"rxclassDrugInfo": []}}
        response = _response()
        response["rxclassDrugInfoList"]["rxclassDrugInfo"][0]["relaSource"] = "FDASPL"
        return response

    memberships = RxClassClient(transport=transport).get_classes("121243")

    assert len(memberships) == 1
    assert memberships[0].relation_source == "FDASPL"


def test_empty_valid_response_is_not_an_error() -> None:
    client = RxClassClient(
        transport=lambda *_: {"rxclassDrugInfoList": {"rxclassDrugInfo": []}}
    )

    assert client.get_classes("121243") == ()


def test_real_empty_object_response_is_not_an_error() -> None:
    client = RxClassClient(transport=lambda *_: {})

    assert client.get_classes("121243") == ()


@pytest.mark.parametrize(
    "exception",
    [
        HTTPError("https://rxnav.nlm.nih.gov", 429, "rate", None, None),
        HTTPError("https://rxnav.nlm.nih.gov", 503, "down", None, None),
        URLError("offline"),
        TimeoutError(),
    ],
)
def test_upstream_failure_is_safe(exception: Exception) -> None:
    client = RxClassClient(
        transport=lambda *_: (_ for _ in ()).throw(exception)
    )

    with pytest.raises(RxClassSourceError):
        client.get_classes("121243")


def test_invalid_input_and_response_are_rejected() -> None:
    client = RxClassClient(transport=lambda *_: {"unexpected": []})

    with pytest.raises(ValueError):
        client.get_classes("not-an-id")
    with pytest.raises(RxClassSourceError, match="invalid_response"):
        client.get_classes("121243")
