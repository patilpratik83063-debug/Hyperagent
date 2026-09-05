"""Weighted quality score + hard acceptance gates (§8).

A candidate must clear EVERY gate — overall >= min, service match >= min,
intent at/above the minimum level, a real buying intent, and the model's own
is_qualified verdict. Clearing the average is not enough.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from models import IntentStrength, LeadClassification, LeadType

VALID_LEAD_TYPES = {t.value for t in LeadType}

# Weights (tunable, shape per §8).
W_INTENT = 0.30
W_SERVICE_MATCH = 0.25
W_COMMERCIAL = 0.15
W_DECISION_MAKER = 0.10
W_LOCATION = 0.10
W_EVIDENCE = 0.10


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    min_overall: float = 60.0
    min_service_match: float = 50.0
    min_intent_strength: str = IntentStrength.RECOMMENDATION.value  # at/above this


@dataclass(frozen=True, slots=True)
class QualifiedLead:
    keep: bool
    overall: float
    service_match: float
    intent_strength: str
    gates: dict[str, bool]
    reason: str

    @property
    def gate_summary(self) -> str:
        passed = ", ".join(k for k, v in self.gates.items() if v) or "none"
        failed = ", ".join(k for k, v in self.gates.items() if not v) or "none"
        return f"passed=[{passed}] failed=[{failed}]"


def intent_rank(strength: str) -> int:
    try:
        return IntentStrength(strength).rank()
    except ValueError:
        return 0


def min_intent_rank(cfg: ScoreConfig) -> int:
    return intent_rank(cfg.min_intent_strength)


def location_confidence(requested_country: str, text: str, evidence: str) -> float:
    """Heuristic: no country requested -> no mismatch risk (100). Country
    requested -> 100 only when it actually appears; otherwise neutral 50
    (we simply cannot verify). Never rewards a mismatch."""
    country = (requested_country or "").strip().lower()
    if not country:
        return 100.0
    haystack = f"{text or ''} {evidence or ''}".lower()
    tokens = [t for t in re.split(r"[\s,.-]+", country) if len(t) > 2]
    if not tokens:
        return 50.0
    return 100.0 if any(t in haystack for t in tokens) else 50.0


def compute_score(
    classification: LeadClassification,
    *,
    requested_country: str = "",
    post_text: str = "",
    cfg: ScoreConfig | None = None,
) -> QualifiedLead:
    cfg = cfg or ScoreConfig()
    rank = intent_rank(classification.intent_strength)
    loc = location_confidence(requested_country, post_text, classification.evidence)

    overall = (
        W_INTENT * rank
        + W_SERVICE_MATCH * classification.service_match_score
        + W_COMMERCIAL * classification.commercial_intent_score
        + W_DECISION_MAKER * (100.0 if classification.decision_maker_signal else 0.0)
        + W_LOCATION * loc
        + W_EVIDENCE * classification.evidence_strength
    )

    gates = {
        "type_is_buyer": classification.lead_type in VALID_LEAD_TYPES,
        "buying_not_selling": (
            classification.is_buying_sourcing
            and not classification.is_selling_offering
            and not classification.is_job_seek
        ),
        "service_match_ok": classification.service_match_score >= cfg.min_service_match,
        "intent_ok": rank >= min_intent_rank(cfg),
        "overall_ok": overall >= cfg.min_overall,
        "model_qualified": classification.is_qualified,
    }
    keep = all(gates.values())
    if not keep:
        failed = [name for name, ok in gates.items() if not ok]
        reason = (
            f"rejected by gate(s): {', '.join(failed)} "
            f"(overall={overall:.0f}, service={classification.service_match_score:.0f}, "
            f"intent={classification.intent_strength}, confidence={classification.confidence:.2f})"
        )
    else:
        reason = (
            f"accepted ({classification.reason}) overall={overall:.0f} "
            f"intent={classification.intent_strength} match={classification.service_match_score:.0f}"
        )
    return QualifiedLead(
        keep=keep,
        overall=round(overall, 1),
        service_match=round(classification.service_match_score, 1),
        intent_strength=classification.intent_strength,
        gates=gates,
        reason=reason,
    )
