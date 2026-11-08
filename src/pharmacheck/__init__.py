"""Core contracts for PharmaCheck AI."""

from pharmacheck.contracts import AnalysisResult, EvidencePackage
from pharmacheck.drug_recognition import DrugCatalog, parse_drug_pair
from pharmacheck.evidence import build_evidence_package, collect_evidence_package
from pharmacheck.telegram import TelegramBotClient, format_analysis_result
from pharmacheck.telegram_webhook import TelegramWebhook
from pharmacheck.yandexgpt import YandexGPTAnalysis, YandexGPTClient, YandexGPTError

__all__ = [
    "AnalysisResult",
    "DrugCatalog",
    "EvidencePackage",
    "build_evidence_package",
    "collect_evidence_package",
    "format_analysis_result",
    "parse_drug_pair",
    "TelegramBotClient",
    "TelegramWebhook",
    "YandexGPTAnalysis",
    "YandexGPTClient",
    "YandexGPTError",
]
