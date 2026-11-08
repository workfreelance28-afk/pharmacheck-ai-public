from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

import pytest

from pharmacheck.contracts import RetrievalStatus, SearchResultStatus
from pharmacheck.sources.dailymed import DailyMedLabelClient
from pharmacheck.sources.rxclass import DrugClassMembership
from pharmacheck.sources.rxnorm import RxNormResolution, RxNormStatus

CHECKED_AT = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
SETID_A = "11111111-1111-1111-1111-111111111111"
SETID_B = "22222222-2222-2222-2222-222222222222"


class FakeRxNorm:
    def __init__(self, resolutions=None):
        self.resolutions = resolutions or {}
        self.calls = []

    def resolve(self, ingredient):
        self.calls.append(ingredient)
        return self.resolutions.get(
            ingredient,
            RxNormResolution(ingredient, RxNormStatus.MATCHED, "5640"),
        )


class FakeRxClass:
    def __init__(self, memberships=None):
        self.memberships = memberships or {}
        self.calls = []

    def get_classes(self, rxcui):
        self.calls.append(rxcui)
        return self.memberships.get(rxcui, ())


def _search_payload(setid: str, title: str) -> bytes:
    return json.dumps(
        {
            "data": [
                {
                    "setid": setid,
                    "title": title,
                    "published_date": "Aug 01, 2026",
                }
            ]
        }
    ).encode()


def _spl_xml(*, counterpart: str | None = None, code: str = "34073-7") -> bytes:
    text = (
        f"Concomitant administration with {counterpart} may increase exposure."
        if counterpart
        else "No pair-specific statement is present in this synthetic fixture."
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <document xmlns="urn:hl7-org:v3">
      <component><structuredBody><component><section>
        <code code="{code}" />
        <title>DRUG INTERACTIONS</title>
        <text><paragraph>{text}</paragraph></text>
      </section></component></structuredBody></component>
    </document>""".encode()


def _nested_spl_xml(*, counterpart: str) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <document xmlns="urn:hl7-org:v3">
      <component><structuredBody><component><section>
        <code code="34070-3" />
        <title>CONTRAINDICATIONS</title>
        <component><section>
          <title>PDE5 Inhibitor Use</title>
          <text><paragraph>Use with {counterpart} is contraindicated.</paragraph></text>
        </section></component>
      </section></component></structuredBody></component>
    </document>""".encode()


def _class_warning_xml() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8"?>
    <document xmlns="urn:hl7-org:v3">
      <component><structuredBody><component><section>
        <code code="34073-7" />
        <title>DRUG INTERACTIONS</title>
        <text><paragraph>
          Concomitant use of QT prolonging drugs including azole antifungals
          should be avoided because it may increase the risk of torsade de pointes.
        </paragraph></text>
      </section></component></structuredBody></component>
    </document>"""


def test_pair_mention_creates_small_citable_document() -> None:
    def transport(url, *_):
        if url.endswith(".xml"):
            return _spl_xml(counterpart="warfarin") if SETID_A in url else _spl_xml()
        query = parse_qs(urlparse(url).query)
        return _search_payload(
            SETID_A if query.get("rxcui") == ["5640"] else SETID_B,
            "Synthetic official label",
        )

    rxnorm = FakeRxNorm(
        {
            "ibuprofen": RxNormResolution(
                "ibuprofen", RxNormStatus.MATCHED, "5640"
            ),
            "warfarin": RxNormResolution(
                "warfarin", RxNormStatus.MATCHED, "11289"
            ),
        }
    )
    outcome = DailyMedLabelClient(rxnorm=rxnorm, transport=transport).search_pair(
        "ibuprofen", "warfarin", checked_at=CHECKED_AT
    )

    assert outcome.complete is True
    assert outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    assert len(outcome.documents) == 1
    document = outcome.documents[0]
    assert document.section == "drug_interactions"
    assert document.retrieval_status is RetrievalStatus.OK
    assert document.document_date.isoformat() == "2026-08-01"
    assert "warfarin" in document.excerpt
    assert len(document.excerpt) <= 4100


def test_pair_mention_in_nested_subsection_is_citable() -> None:
    def transport(url, *_):
        if url.endswith(".xml"):
            return (
                _nested_spl_xml(counterpart="sildenafil")
                if SETID_B in url
                else _spl_xml()
            )
        query = parse_qs(urlparse(url).query)
        return _search_payload(
            SETID_A if query.get("rxcui") == ["136411"] else SETID_B,
            "Synthetic official label",
        )

    rxnorm = FakeRxNorm(
        {
            "sildenafil": RxNormResolution(
                "sildenafil", RxNormStatus.MATCHED, "136411"
            ),
            "nitroglycerin": RxNormResolution(
                "nitroglycerin", RxNormStatus.MATCHED, "4917"
            ),
        }
    )

    outcome = DailyMedLabelClient(rxnorm=rxnorm, transport=transport).search_pair(
        "sildenafil", "nitroglycerin", checked_at=CHECKED_AT
    )

    assert outcome.complete is True
    assert outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    assert len(outcome.documents) == 1
    assert outcome.documents[0].section == "contraindications"
    assert "sildenafil" in outcome.documents[0].excerpt


def test_structured_counterpart_class_links_label_warning_to_pair() -> None:
    def transport(url, *_):
        if url.endswith(".xml"):
            return _class_warning_xml() if SETID_A in url else _spl_xml()
        query = parse_qs(urlparse(url).query)
        return _search_payload(
            SETID_A if query.get("rxcui") == ["703"] else SETID_B,
            "Synthetic official label",
        )

    rxnorm = FakeRxNorm(
        {
            "amiodarone": RxNormResolution(
                "amiodarone", RxNormStatus.MATCHED, "703"
            ),
            "voriconazole": RxNormResolution(
                "voriconazole", RxNormStatus.MATCHED, "121243"
            ),
        }
    )
    rxclass = FakeRxClass(
        {
            "121243": (
                DrugClassMembership(
                    rxcui="121243",
                    class_id="D001393",
                    class_name="Azoles",
                    class_type="CHEM",
                    relation="has_chemical_structure",
                    relation_source="DailyMed",
                ),
            )
        }
    )

    outcome = DailyMedLabelClient(
        rxnorm=rxnorm,
        rxclass=rxclass,
        transport=transport,
    ).search_pair("amiodarone", "voriconazole", checked_at=CHECKED_AT)

    assert outcome.complete is True
    assert outcome.search_result.result is SearchResultStatus.INTERACTION_FOUND
    assert len(outcome.documents) == 1
    document = outcome.documents[0]
    assert document.match_basis == "structured_class"
    assert document.matched_term == "Azoles"
    assert document.matched_ingredient == "voriconazole"
    assert "azole antifungals" in document.excerpt.casefold()
    assert "rxcui=121243" in str(document.additional_urls[0])


def test_clean_two_direction_search_is_negative_not_safety_claim() -> None:
    def transport(url, *_):
        if url.endswith(".xml"):
            return _spl_xml()
        return _search_payload(SETID_A, "Synthetic label")

    rxnorm = FakeRxNorm(
        {
            "alpha": RxNormResolution("alpha", RxNormStatus.MATCHED, "111"),
            "beta": RxNormResolution("beta", RxNormStatus.MATCHED, "222"),
        }
    )
    outcome = DailyMedLabelClient(rxnorm=rxnorm, transport=transport).search_pair(
        "alpha", "beta", checked_at=CHECKED_AT
    )

    assert outcome.complete is True
    assert outcome.documents == []
    assert (
        outcome.search_result.result
        is SearchResultStatus.NO_RELEVANT_INTERACTION_FOUND
    )
    assert "безопас" not in (outcome.search_result.details or "").casefold()
    assert "rxcui=111" in str(outcome.search_result.url)
    assert [str(url) for url in outcome.search_result.additional_urls] == [
        "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?pagesize=2&page=1&rxcui=222"
    ]


def test_unmatched_rxnorm_uses_generic_name_search() -> None:
    seen_queries = []

    def transport(url, *_):
        if url.endswith(".xml"):
            return _spl_xml()
        seen_queries.append(parse_qs(urlparse(url).query))
        return _search_payload(SETID_A, "Synthetic label")

    rxnorm = FakeRxNorm(
        {
            "alpha": RxNormResolution("alpha", RxNormStatus.NOT_FOUND),
            "beta": RxNormResolution("beta", RxNormStatus.AMBIGUOUS),
        }
    )
    outcome = DailyMedLabelClient(rxnorm=rxnorm, transport=transport).search_pair(
        "alpha", "beta", checked_at=CHECKED_AT
    )

    assert outcome.complete is True
    assert all("drug_name" in query and "rxcui" not in query for query in seen_queries)


def test_missing_label_is_inconclusive() -> None:
    def transport(url, *_):
        if url.endswith(".xml"):
            pytest.fail("XML must not be fetched")
        return json.dumps({"data": []}).encode()

    outcome = DailyMedLabelClient(
        rxnorm=FakeRxNorm(), transport=transport
    ).search_pair("alpha", "beta", checked_at=CHECKED_AT)

    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.INCONCLUSIVE


def test_network_failure_never_becomes_negative() -> None:
    outcome = DailyMedLabelClient(
        rxnorm=FakeRxNorm(),
        transport=lambda *_: (_ for _ in ()).throw(URLError("offline")),
    ).search_pair("alpha", "beta", checked_at=CHECKED_AT)

    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.FAILED
    assert outcome.failed_queries


def test_malformed_xml_is_failed() -> None:
    def transport(url, *_):
        if url.endswith(".xml"):
            return b"<not-closed"
        return _search_payload(SETID_A, "Synthetic label")

    outcome = DailyMedLabelClient(
        rxnorm=FakeRxNorm(), transport=transport
    ).search_pair("alpha", "beta", checked_at=CHECKED_AT)

    assert outcome.complete is False
    assert outcome.search_result.result is SearchResultStatus.FAILED


@pytest.mark.parametrize("value", ["", "  ", "bad\nvalue", "x" * 201])
def test_invalid_ingredient_is_rejected(value: str) -> None:
    client = DailyMedLabelClient(
        rxnorm=FakeRxNorm(), transport=lambda *_: pytest.fail("network called")
    )

    with pytest.raises(ValueError):
        client.search_pair(value, "valid")
