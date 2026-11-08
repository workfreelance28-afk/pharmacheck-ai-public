import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from pharmacheck.contracts import (
    AnalysisResult,
    InteractionLevel,
    replace_unsafe_recommended_action,
)
from pharmacheck.eval_cases import build_stage1_cases
from pharmacheck.yandexgpt import YandexGPTClient, YandexGPTError


def _green_evidence():
    return next(
        case.evidence
        for case in build_stage1_cases()
        if case.case_id == "green_complete_search"
    )


def _orange_evidence():
    return next(
        case.evidence
        for case in build_stage1_cases()
        if case.case_id == "orange_clinically_significant"
    )


def _green_result() -> dict:
    return {
        "status": "answer",
        "interaction_level": "green",
        "headline": "Значимого известного взаимодействия не найдено",
        "identified_drugs": [
            {"trade_name": "Тестпрепарат Альфа", "active_ingredients": ["вещество альфа"]},
            {"trade_name": "Тестпрепарат Бета", "active_ingredients": ["вещество бета"]},
        ],
        "summary": "В проверенных источниках значимого известного взаимодействия не "
        "найдено, но это не доказывает безопасность комбинации для конкретного человека.",
        "mechanism_or_effect": None,
        "recommended_user_action": "Не изменяйте лечение без консультации врача.",
        "limitations": ["Индивидуальные факторы не оценивались."],
        "conflicts": [],
        "missing_information": [],
        "sources": [
            {
                "source_id": "official.alpha",
                "title": "Официальная инструкция Альфа",
                "url": "https://example.test/official.alpha",
                "supports": "Поиск не выявил прямого упоминания взаимодействия.",
            },
            {
                "source_id": "official.beta",
                "title": "Официальная инструкция Бета",
                "url": "https://example.test/official.beta",
                "supports": "Поиск не выявил прямого упоминания взаимодействия.",
            },
        ],
    }


def _client(transport, *, token="secret-token") -> YandexGPTClient:
    return YandexGPTClient(
        model_uri="gpt://folder/yandexgpt-5.1",
        token_provider=lambda: token,
        system_prompt="system rules",
        folder_id="folder",
        transport=transport,
    )


def test_valid_response_is_parsed_and_usage_is_returned():
    def transport(request, timeout):
        assert timeout == 60.0
        assert request.get_header("Authorization") == "Bearer secret-token"
        payload = json.loads(request.data)
        assert payload["model"] == "gpt://folder/yandexgpt-5.1"
        assert payload["text"]["format"]["type"] == "json_schema"
        assert "deterministic_assessment" in json.loads(payload["input"])
        return {
            "id": "response-1",
            "output_text": json.dumps(_green_result(), ensure_ascii=False),
            "usage": {"input_tokens": 1200, "output_tokens": "300"},
        }

    analysis = _client(transport).analyze(_green_evidence())

    assert analysis.result.interaction_level.value == "green"
    assert analysis.usage.total_tokens == 1500
    assert analysis.response_id == "response-1"


def test_nested_output_is_supported():
    response = {
        "output": [
            {
                "content": [
                    {"type": "output_text", "text": json.dumps(_green_result())}
                ]
            }
        ]
    }
    assert _client(lambda request, timeout: response).analyze(_green_evidence()).result


def test_disallowed_level_is_rejected_after_model_response():
    result = _green_result()
    result["interaction_level"] = "red"
    response = {"output_text": json.dumps(result)}

    with pytest.raises(YandexGPTError, match="защитную проверку") as error:
        _client(lambda request, timeout: response).analyze(_green_evidence())

    assert error.value.code == "invalid_model_response"
    assert not error.value.retryable


def test_unsupported_red_is_downgraded_to_orange_without_retry():
    result = _green_result()
    result.update(
        {
            "interaction_level": "red",
            "headline": "Обнаружено потенциально значимое взаимодействие",
            "summary": "Источники описывают клинически значимый риск.",
            "mechanism_or_effect": "Возможен клинически значимый эффект.",
        }
    )
    calls = 0

    def transport(request, timeout):
        nonlocal calls
        calls += 1
        return {"output_text": json.dumps(result, ensure_ascii=False)}

    analysis = _client(transport).analyze(_orange_evidence())

    assert calls == 1
    assert analysis.result.interaction_level is InteractionLevel.ORANGE
    assert analysis.result.recommended_user_action == (
        "Обсудите результат с врачом или фармацевтом. "
        "Не изменяйте лечение самостоятельно."
    )


@pytest.mark.parametrize(
    "action",
    [
        "Немедленно прекратите прием одного из препаратов и свяжитесь с врачом.",
        "Немедленно прекратите совместный приём препаратов и свяжитесь с врачом.",
        "Прекратите одновременный прием этих лекарств.",
        "Отмените препарат до консультации.",
        "Замените лекарство самостоятельно.",
        "Уменьшите дозировку в два раза.",
        "Stop taking the medication and call your doctor.",
    ],
)
def test_self_directed_treatment_actions_are_replaced(action):
    result = _green_result()
    result["recommended_user_action"] = action
    response = {"output_text": json.dumps(result, ensure_ascii=False)}

    analysis = _client(lambda request, timeout: response).analyze(_green_evidence())

    assert analysis.result.recommended_user_action == (
        "Обсудите результат с врачом или фармацевтом. "
        "Не изменяйте лечение самостоятельно."
    )


def test_self_directed_treatment_instruction_outside_action_is_rejected():
    result = _green_result()
    result["summary"] = "Немедленно прекратите прием одного из препаратов."
    response = {"output_text": json.dumps(result, ensure_ascii=False)}

    with pytest.raises(YandexGPTError, match="защитную проверку") as error:
        _client(lambda request, timeout: response).analyze(_green_evidence())

    assert error.value.code == "invalid_model_response"


def test_safe_model_action_is_still_replaced_by_deterministic_template():
    result = _green_result()
    result["recommended_user_action"] = (
        "Не прекращайте и не изменяйте лечение без консультации врача."
    )
    response = {"output_text": json.dumps(result, ensure_ascii=False)}

    analysis = _client(lambda request, timeout: response).analyze(_green_evidence())

    assert analysis.result.recommended_user_action == (
        "Обсудите результат с врачом или фармацевтом. "
        "Не изменяйте лечение самостоятельно."
    )


def test_red_model_action_is_always_replaced_by_deterministic_template():
    result = AnalysisResult.model_validate(_green_result()).model_copy(
        update={
            "interaction_level": InteractionLevel.RED,
            "recommended_user_action": "Свяжитесь с врачом.",
        }
    )

    safe = replace_unsafe_recommended_action(result)

    assert "Как можно скорее свяжитесь" in safe.recommended_user_action
    assert "Не изменяйте лечение самостоятельно" in safe.recommended_user_action


def test_red_unsafe_action_uses_urgent_clinician_template():
    result = AnalysisResult.model_validate(_green_result()).model_copy(
        update={
            "interaction_level": InteractionLevel.RED,
            "recommended_user_action": "Немедленно прекратите прием препарата.",
        }
    )

    safe = replace_unsafe_recommended_action(result)

    assert "Как можно скорее свяжитесь" in safe.recommended_user_action
    assert "Не изменяйте лечение самостоятельно" in safe.recommended_user_action
    assert "прекратите прием" not in safe.recommended_user_action.casefold()


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [(401, "authentication_error", False), (429, "rate_limited", True), (503, "service_unavailable", True)],
)
def test_http_errors_are_sanitized(status, code, retryable):
    def transport(request, timeout):
        raise HTTPError(request.full_url, status, "sensitive body", {}, BytesIO(b"secret"))

    with pytest.raises(YandexGPTError) as error:
        _client(transport).analyze(_green_evidence())

    assert error.value.code == code
    assert error.value.retryable is retryable
    assert "secret" not in str(error.value)


def test_missing_output_is_a_safe_invalid_response():
    with pytest.raises(YandexGPTError) as error:
        _client(lambda request, timeout: {}).analyze(_green_evidence())
    assert error.value.code == "invalid_model_response"


def test_missing_token_stops_before_transport():
    with pytest.raises(YandexGPTError) as error:
        _client(lambda request, timeout: pytest.fail("must not call"), token="").analyze(
            _green_evidence()
        )
    assert error.value.code == "missing_iam_token"


def test_missing_deterministic_assessment_stops_before_transport():
    raw = _green_evidence().model_dump(mode="json")
    raw["deterministic_assessment"] = None
    evidence = type(_green_evidence()).model_validate(raw)

    with pytest.raises(YandexGPTError) as error:
        _client(lambda request, timeout: pytest.fail("must not call")).analyze(evidence)
    assert error.value.code == "missing_assessment"
