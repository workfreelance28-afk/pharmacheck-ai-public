"""External medical-source adapters.

`pharmacheck.sources.ddinter` is deliberately not imported here. This
package `__init__` must import cleanly in an `official_labels_demo`
deployment archive, which omits `ddinter.py` entirely (D-011). Code that
needs `DDInterSource` imports it directly from
`pharmacheck.sources.ddinter`, which only happens on the
`noncommercial_mvp` path (see `cloud_runtime._build_sources`).
"""

from pharmacheck.sources.base import InteractionSearchOutcome, InteractionSource
from pharmacheck.sources.dailymed import DailyMedLabelClient, DailyMedSearchOutcome
from pharmacheck.sources.openfda import OpenFDALabelClient, OpenFDASearchOutcome
from pharmacheck.sources.rxnorm import (
    RxNormClient,
    RxNormResolution,
    RxNormSourceError,
    RxNormStatus,
)

__all__ = [
    "DailyMedLabelClient",
    "DailyMedSearchOutcome",
    "InteractionSearchOutcome",
    "InteractionSource",
    "OpenFDALabelClient",
    "OpenFDASearchOutcome",
    "RxNormClient",
    "RxNormResolution",
    "RxNormSourceError",
    "RxNormStatus",
]
