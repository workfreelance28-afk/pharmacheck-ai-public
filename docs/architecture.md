# PharmaCheck AI — architecture

## Active non-commercial production flow

```text
Telegram user
  ↓
Cloudflare Worker relay
  ↓
Yandex Cloud Function (Python 3.12)
  ↓
GRLS-derived local catalog
  ↓
RxNorm identity and RxCUI
  ↓
DailyMed official labels + RxClass structured classes
  ↓
Local DDInter 2.0 snapshot (non-commercial MVP)
  ↓
Deterministic evidence assessment
  ↓
Deterministic result for simple cases or YandexGPT 5.1 for evidence analysis
  ↓
Strict output validation and Telegram formatting
```

## Component responsibilities

- `drug_recognition.py`: parse two drug mentions and resolve trusted catalog identities.
- `sources/rxnorm.py`: verify an English ingredient and obtain RxCUI.
- `sources/dailymed.py`: retrieve selected official SPL labels and search relevant sections.
- `sources/rxclass.py`: obtain direct structured class memberships for a resolved RxCUI.
- `sources/ddinter.py`: query the verified local non-commercial DDInter snapshot.
- `evidence.py`: assemble coverage, documents, conflicts, and allowed levels.
- `application.py`: stop unsafe cases before the model and coordinate the workflow.
- `yandexgpt.py`: request strict schema output and reject unsafe or unsupported results.
- `telegram.py` and `telegram_webhook.py`: delivery, formatting, secret validation, and deduplication.
- `cloud_runtime.py`: Yandex-specific configuration, IAM token acquisition, and handler assembly.

## Trust boundaries

- Drug identity must come from the verified catalog and RxNorm, not model inference.
- Medical conclusions must be traceable to retrieved source documents.
- Search-result absence is weaker than positive regulator evidence.
- Red requires explicit official-label avoidance or contraindication evidence.
- Model text is untrusted until schema and evidence validation pass.
- Bot tokens, webhook secrets, and IAM credentials remain outside Git in Yandex Lockbox.

## Deployment boundaries

- Telegram enters through the Cloudflare relay.
- Medical processing and YandexGPT access run in Yandex Cloud.
- The deployment ZIP is generated locally and excluded from Git.
- DDInter CSV files are excluded from Git and restored separately for authorized non-commercial use.


## Client-demo deployment (published 2026-08-13, frozen per D-016)

The client demo is a second deployment from the same repository. It is not a
fork or a copy of the project, and it does not replace the active bot.

```text
Client-demo Telegram bot
  ↓
Dedicated Cloudflare webhook route
  ↓
Dedicated Yandex Cloud Function (Python 3.12)
  ↓
GRLS-derived verified pilot catalog
  ↓
RxNorm identity and RxCUI
  ↓
DailyMed + openFDA label retrieval + RxClass structured classes
  ↓
Deterministic evidence assessment
  ↓
Deterministic result or YandexGPT analysis of retrieved evidence
  ↓
Strict validation, safe action text, source links, and Telegram formatting
```

Configuration modes:

- `noncommercial_mvp`: current private prototype, including the separately
  restored DDInter snapshot;
- `official_labels_demo`: client-facing demo with no DDInter data or runtime
  calls.

Deployment isolation requirements:

- separate Telegram bot token and webhook secret;
- separate Cloud Function and webhook route;
- no DDInter CSV in the demo archive;
- no shared Telegram update processing between the two bots;
- the active production version is not changed while the demo is developed;
- DailyMed and openFDA results are deduplicated by label identity where
  possible and are not described as independent databases.

The demo additionally uses dedicated Cloudflare Workers (D-014) and
privacy-minimizing analytics (D-012). Private operations material is excluded from this public source release.
