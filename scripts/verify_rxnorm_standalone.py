"""Verify a Qwen candidate batch against official RxNorm using only stdlib."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
PROPERTIES_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json"
RELATED_URL = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json?tty=IN"
SOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9 .'-]{0,199}$")


class VerificationError(RuntimeError):
    """A safe, classified verification failure."""


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def fetch_json(url: str, *, timeout: float, attempts: int = 3) -> dict[str, Any]:
    request = Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "PharmaCheckAI/0.1"},
        method="GET",
    )
    last_error = "network_error"
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read(500_001)
            if len(payload) > 500_000:
                raise VerificationError("response_too_large")
            decoded = json.loads(payload.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise VerificationError("invalid_json_root")
            return decoded
        except HTTPError as exc:
            if exc.code == 429:
                last_error = "rate_limited"
            elif 500 <= exc.code <= 599:
                last_error = "upstream_unavailable"
            else:
                raise VerificationError(f"http_{exc.code}") from exc
        except (TimeoutError, URLError):
            last_error = "network_error"
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise VerificationError("invalid_json") from exc
        if attempt < attempts - 1:
            time.sleep(0.5 * (2**attempt))
    raise VerificationError(last_error)


def load_candidates(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("candidate file has an unsupported schema")
    if raw.get("generated_by") != "qwen" or not isinstance(raw.get("candidates"), list):
        raise ValueError("candidate file is invalid")
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw["candidates"]:
        if not isinstance(item, dict):
            raise ValueError("candidate item is invalid")
        russian_name = item.get("russian_name")
        source_name = item.get("source_name_candidate")
        if not isinstance(russian_name, str) or not russian_name.strip():
            raise ValueError("candidate Russian name is invalid")
        if not isinstance(source_name, str) or not SOURCE_NAME_PATTERN.fullmatch(source_name):
            raise ValueError("candidate source name is invalid")
        normalized = normalize_name(russian_name).replace("ё", "е")
        if normalized in seen:
            raise ValueError("candidate batch contains duplicate Russian names")
        seen.add(normalized)
        candidates.append(
            {"russian_name": russian_name.strip(), "source_name_candidate": source_name}
        )
    if not candidates:
        raise ValueError("candidate batch is empty")
    return candidates


def resolve(source_name: str, *, timeout: float) -> tuple[str | None, str]:
    query = urlencode({"name": source_name, "search": "2", "allsrc": "0"})
    response = fetch_json(f"{RXCUI_URL}?{query}", timeout=timeout)
    group = response.get("idGroup")
    if not isinstance(group, dict):
        raise VerificationError("invalid_response")
    identifiers = group.get("rxnormId") or []
    if not isinstance(identifiers, list) or any(
        not isinstance(value, str) or not value.isdigit() for value in identifiers
    ):
        raise VerificationError("invalid_response")
    identifiers = list(dict.fromkeys(identifiers))
    if not identifiers:
        return None, "not_found"
    if len(identifiers) > 1:
        return None, "ambiguous"
    return identifiers[0], "matched"


def get_properties(rxcui: str, *, timeout: float) -> dict[str, str]:
    response = fetch_json(PROPERTIES_URL.format(rxcui=rxcui), timeout=timeout)
    properties = response.get("properties")
    required = ("rxcui", "name", "tty", "language", "suppress")
    if not isinstance(properties, dict) or any(
        not isinstance(properties.get(key), str) for key in required
    ):
        raise VerificationError("invalid_response")
    if properties["rxcui"] != rxcui or not properties["name"].strip():
        raise VerificationError("invalid_response")
    return {key: properties[key] for key in required}


def rejection_reason(source_name: str, properties: dict[str, str]) -> str | None:
    if normalize_name(source_name) != normalize_name(properties["name"]):
        return "preferred_name_mismatch"
    if properties["tty"] != "IN":
        return "not_ingredient"
    if properties["language"] != "ENG":
        return "not_english"
    if properties["suppress"] != "N":
        return "suppressed"
    return None


def get_related_ingredient(rxcui: str, *, timeout: float) -> dict[str, str] | None:
    """Return the single official `IN` concept related to a `PIN` rxcui, if any.

    Keeps stored ingredient names consistent with the `IN`-level vocabulary
    that DDInter matching relies on, instead of accepting a precise salt
    form (`PIN`) as-is.
    """
    response = fetch_json(RELATED_URL.format(rxcui=rxcui), timeout=timeout)
    group = response.get("relatedGroup")
    if not isinstance(group, dict):
        raise VerificationError("invalid_response")
    concept_groups = group.get("conceptGroup")
    if concept_groups is None:
        return None
    if not isinstance(concept_groups, list):
        raise VerificationError("invalid_response")
    required = ("rxcui", "name", "tty", "language", "suppress")
    concepts: list[dict[str, str]] = []
    for concept_group in concept_groups:
        if not isinstance(concept_group, dict) or concept_group.get("tty") != "IN":
            continue
        properties = concept_group.get("conceptProperties") or []
        if not isinstance(properties, list):
            raise VerificationError("invalid_response")
        for item in properties:
            if not isinstance(item, dict) or any(
                not isinstance(item.get(key), str) for key in required
            ):
                raise VerificationError("invalid_response")
            concepts.append({key: item[key] for key in required})
    if len(concepts) != 1:
        return None
    return concepts[0]


def verify(path: Path, *, timeout: float, delay: float) -> dict[str, Any]:
    candidates = load_candidates(path)
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    resolved_from_precise_ingredient: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        russian_name = candidate["russian_name"]
        source_name = candidate["source_name_candidate"]
        try:
            rxcui, status = resolve(source_name, timeout=timeout)
            if rxcui is None:
                rejected.append({**candidate, "reason": status})
            else:
                properties = get_properties(rxcui, timeout=timeout)
                reason = rejection_reason(source_name, properties)
                if reason == "not_ingredient" and properties["tty"] == "PIN":
                    parent = get_related_ingredient(rxcui, timeout=timeout)
                    if (
                        parent is not None
                        and parent["language"] == "ENG"
                        and parent["suppress"] == "N"
                    ):
                        accepted.append(
                            {
                                "russian_name": russian_name,
                                "source_name": parent["name"],
                                "source_id": f"rxnorm.rxcui.{parent['rxcui']}",
                                "source_url": PROPERTIES_URL.format(rxcui=parent["rxcui"]),
                            }
                        )
                        resolved_from_precise_ingredient.append(
                            {
                                "russian_name": russian_name,
                                "source_name_candidate": source_name,
                                "precise_ingredient_name": properties["name"],
                                "precise_ingredient_rxcui": rxcui,
                                "resolved_ingredient_name": parent["name"],
                                "resolved_ingredient_rxcui": parent["rxcui"],
                            }
                        )
                        if delay and index < len(candidates) - 1:
                            time.sleep(delay)
                        continue
                if reason:
                    rejected.append(
                        {
                            **candidate,
                            "reason": reason,
                            "official_name": properties["name"],
                            "rxcui": rxcui,
                        }
                    )
                else:
                    accepted.append(
                        {
                            "russian_name": russian_name,
                            "source_name": properties["name"],
                            "source_id": f"rxnorm.rxcui.{rxcui}",
                            "source_url": PROPERTIES_URL.format(rxcui=rxcui),
                        }
                    )
        except VerificationError as exc:
            errors.append({**candidate, "error": str(exc)})
        if delay and index < len(candidates) - 1:
            time.sleep(delay)
    return {
        "schema_version": 1,
        "verified_against": "RxNorm API",
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "error_count": len(errors),
        "accepted": accepted,
        "rejected": rejected,
        "errors": errors,
        "resolved_from_precise_ingredient": resolved_from_precise_ingredient,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()
    if args.timeout <= 0 or args.delay < 0:
        parser.error("timeout and delay must be valid")
    report = verify(args.candidates, timeout=args.timeout, delay=args.delay)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: report[key] for key in ("candidate_count", "accepted_count", "rejected_count", "error_count")},
            ensure_ascii=False,
        )
    )
    return 0 if report["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
