"""Run synthetic stage-1 cases against Yandex AI Studio Responses API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from pharmacheck.contracts import (  # noqa: E402
    AnalysisResult,
    ContractViolation,
    validate_analysis_against_evidence,
)
from pharmacheck.eval_cases import EvalCase, build_stage1_cases  # noqa: E402
from pharmacheck.yandexgpt import YandexGPTClient, YandexGPTError  # noqa: E402

PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt_v1.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Выполнить платные API-вызовы")
    parser.add_argument("--case", default="all", help="ID сценария или all")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cases = select_cases(build_stage1_cases(), args.case)

    if not args.execute:
        print("Dry run: API не вызывается. Доступные сценарии:")
        for case in cases:
            print(f"- {case.case_id}: {case.description}")
        print("Для реального запуска добавьте --execute и задайте облачные переменные.")
        return 0

    iam_token = require_env("YC_IAM_TOKEN")
    model_uri = require_env("YANDEXGPT_MODEL_URI")
    folder_id = os.getenv("YC_FOLDER_ID")
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    client = YandexGPTClient(
        model_uri=model_uri,
        token_provider=lambda: iam_token,
        system_prompt=system_prompt,
        folder_id=folder_id,
        timeout=args.timeout,
    )

    failures = 0
    input_tokens = 0
    output_tokens = 0
    for case in cases:
        try:
            analysis = client.analyze(case.evidence)
            validate_case(case, analysis.result)
            input_tokens += analysis.usage.input_tokens or 0
            output_tokens += analysis.usage.output_tokens or 0
            print(
                f"PASS {case.case_id}: {analysis.result.status.value}/"
                f"{analysis.result.interaction_level.value}"
            )
        except (ValueError, ContractViolation, YandexGPTError) as exc:
            failures += 1
            print(f"FAIL {case.case_id}: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(f"Итого: {len(cases) - failures}/{len(cases)} сценариев пройдено")
    if input_tokens or output_tokens:
        print(f"Токены: вход {input_tokens}, выход {output_tokens}")
    return 1 if failures else 0


def select_cases(cases: tuple[EvalCase, ...], case_id: str) -> tuple[EvalCase, ...]:
    if case_id == "all":
        return cases
    selected = tuple(case for case in cases if case.case_id == case_id)
    if not selected:
        raise ValueError(f"Неизвестный сценарий: {case_id}")
    return selected


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Не задана переменная {name}")
    return value


def validate_case(case: EvalCase, result: AnalysisResult) -> None:
    validate_analysis_against_evidence(case.evidence, result)
    if result.status is not case.expected.status:
        raise ValueError(
            f"expected status {case.expected.status.value}, got {result.status.value}"
        )
    if result.interaction_level is not case.expected.level:
        raise ValueError(
            f"expected level {case.expected.level.value}, "
            f"got {result.interaction_level.value}"
        )

    answer_text = " ".join(
        [result.headline, result.summary, result.recommended_user_action]
    ).casefold()
    for fragment in case.expected.forbidden_fragments:
        if fragment.casefold() in answer_text:
            raise ValueError(f"forbidden recommendation found: {fragment}")


if __name__ == "__main__":
    raise SystemExit(main())
