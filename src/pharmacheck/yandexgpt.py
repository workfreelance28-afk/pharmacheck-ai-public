"""Validated Yandex AI Studio integration for the analysis stage."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from pharmacheck.contracts import (
    AnalysisResult,
    AnalysisStatus,
    ContractViolation,
    EvidencePackage,
    InteractionLevel,
    replace_unsafe_recommended_action,
    validate_analysis_against_evidence,
)

DEFAULT_RESPONSES_URL = "https://ai.api.cloud.yandex.net/v1/responses"


class YandexGPTError(RuntimeError):
    """Safe error that can be logged without exposing credentials or response bodies."""

    def __init__(self, message: str, *, code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class YandexGPTUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class YandexGPTAnalysis:
    result: AnalysisResult
    usage: YandexGPTUsage
    response_id: str | None = None


Transport = Callable[[Request, float], Mapping[str, Any]]


class YandexGPTClient:
    """Send an evidence package and accept only a schema-valid, safe answer."""

    def __init__(
        self,
        *,
        model_uri: str,
        token_provider: Callable[[], str],
        system_prompt: str,
        folder_id: str | None = None,
        endpoint: str = DEFAULT_RESPONSES_URL,
        timeout: float = 60.0,
        max_output_tokens: int = 2500,
        transport: Transport | None = None,
    ) -> None:
        if not model_uri.startswith("gpt://"):
            raise ValueError("model_uri must be an explicit gpt:// URI")
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if timeout <= 0 or max_output_tokens <= 0:
            raise ValueError("timeout and max_output_tokens must be positive")

        self._model_uri = model_uri
        self._token_provider = token_provider
        self._system_prompt = system_prompt
        self._folder_id = folder_id
        self._endpoint = endpoint
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens
        self._transport = transport or _default_transport

    def analyze(self, evidence: EvidencePackage) -> YandexGPTAnalysis:
        if evidence.deterministic_assessment is None:
            raise YandexGPTError(
                "Пакет доказательств не прошёл детерминированную оценку.",
                code="missing_assessment",
            )

        token = self._token_provider().strip()
        if not token:
            raise YandexGPTError(
                "Не удалось получить IAM-токен сервисного аккаунта.",
                code="missing_iam_token",
            )

        request = Request(
            self._endpoint,
            data=json.dumps(self._payload(evidence), ensure_ascii=False).encode("utf-8"),
            headers=self._headers(token),
            method="POST",
        )
        try:
            response = self._transport(request, self._timeout)
        except HTTPError as exc:
            raise _http_error(exc.code) from exc
        except TimeoutError as exc:
            raise YandexGPTError(
                "YandexGPT не ответил за отведённое время.",
                code="timeout",
                retryable=True,
            ) from exc
        except URLError as exc:
            raise YandexGPTError(
                "Не удалось связаться с Yandex AI Studio.",
                code="network_error",
                retryable=True,
            ) from exc
        except (UnicodeError, ValueError) as exc:
            raise YandexGPTError(
                "Yandex AI Studio вернул повреждённый ответ.",
                code="invalid_service_response",
                retryable=True,
            ) from exc

        try:
            result = AnalysisResult.model_validate_json(_extract_output_text(response))
            result = _clamp_disallowed_red_to_orange(evidence, result)
            result = replace_unsafe_recommended_action(result)
            validate_analysis_against_evidence(evidence, result)
        except (ValueError, ValidationError, ContractViolation) as exc:
            raise YandexGPTError(
                "YandexGPT вернул ответ, не прошедший защитную проверку.",
                code="invalid_model_response",
            ) from exc

        return YandexGPTAnalysis(
            result=result,
            usage=_extract_usage(response),
            response_id=_optional_string(response.get("id")),
        )

    def _payload(self, evidence: EvidencePackage) -> dict[str, Any]:
        return {
            "model": self._model_uri,
            "instructions": self._system_prompt,
            "input": evidence.model_dump_json(),
            "max_output_tokens": self._max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "pharmacheck_analysis",
                    "description": "Структурированный результат PharmaCheck AI",
                    "schema": AnalysisResult.model_json_schema(mode="serialization"),
                    "strict": True,
                }
            },
        }

    def _headers(self, token: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        if self._folder_id:
            headers["x-folder-id"] = self._folder_id
        return headers


def _clamp_disallowed_red_to_orange(
    evidence: EvidencePackage,
    result: AnalysisResult,
) -> AnalysisResult:
    """Downgrade an unsupported red answer without making a new model call."""
    assessment = evidence.deterministic_assessment
    if (
        result.status is AnalysisStatus.ANSWER
        and result.interaction_level is InteractionLevel.RED
        and assessment is not None
        and InteractionLevel.RED not in assessment.allowed_levels
        and InteractionLevel.ORANGE in assessment.allowed_levels
    ):
        return result.model_copy(update={"interaction_level": InteractionLevel.ORANGE})
    return result


def _default_transport(request: Request, timeout: float) -> Mapping[str, Any]:
    with urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("Yandex AI Studio response must be an object")
    return decoded


def _http_error(status: int) -> YandexGPTError:
    if status in {401, 403}:
        return YandexGPTError(
            "Yandex AI Studio отклонил учётные данные или права доступа.",
            code="authentication_error",
        )
    if status == 429:
        return YandexGPTError(
            "Превышена квота запросов Yandex AI Studio.",
            code="rate_limited",
            retryable=True,
        )
    if status >= 500:
        return YandexGPTError(
            "Yandex AI Studio временно недоступен.",
            code="service_unavailable",
            retryable=True,
        )
    return YandexGPTError(
        "Yandex AI Studio отклонил запрос.",
        code="request_rejected",
    )


def _extract_output_text(response: Mapping[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and part.get("type") in {
                    "output_text",
                    "text",
                }:
                    value = part.get("text")
                    if isinstance(value, str) and value:
                        return value
    raise ValueError("Yandex AI Studio response does not contain output text")


def _extract_usage(response: Mapping[str, Any]) -> YandexGPTUsage:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return YandexGPTUsage()
    input_tokens = _optional_int(usage.get("input_tokens"))
    output_tokens = _optional_int(usage.get("output_tokens"))
    total_tokens = _optional_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return YandexGPTUsage(input_tokens, output_tokens, total_tokens)


def _optional_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
