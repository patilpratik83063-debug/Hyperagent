"""Deterministic cheap rejects — run BEFORE any AI spend.

Precision doctrine: a deterministic rule only drops a post when the drop is
near-certain. Whenever a genuine buyer phrase and a negative category both
appear (genuine buyers do say "we are hiring", "budget", "DM me if..."), the
post is kept for the classifier instead of cheap-dropped. Pure advice,
self-promo, job-seeker and recruiter content with NO buyer phrase is dropped
here for free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Strong phrases proving the author is on the BUY side.
BUYER_MARKERS: tuple[str, ...] = (
    "looking for", "look for a", "need a ", "need an ", "need someone", "need help with",
    "need to find", "anyone know", "anyone recommend", "anyone has a good", "recommendations for",
    "can you recommend", "do you know anyone", "in need of", "searching for", "seeking a ",
    "seeking an ", "looking to hire", "want to hire", "we are looking", "we're looking",
    "our company needs", "our team needs", "our agency needs", "agency needs", "we need",
    "looking for freelancers", "freelance help for", "on the hunt for", "who can handle",
    "who can help", "dm me if", "dm if you know", "send portfolios", "we are hiring a freelance",
    "hiring a freelance", "hiring freelance", "freelancer needed", "freelancers needed",
)

# Categories that are near-certain negatives when NO buyer phrase is present.
JOB_AD_STRONG: tuple[str, ...] = (
    "vacancy", "apply now", "apply here", "full-time position", "full time position",
    "open position", "job opening", "send your resume", "salary range", "benefits package",
    "apply at careers", "careers page",
)

SELLER_MARKERS: tuple[str, ...] = (
    "we offer", "we provide", "we specialize", "our agency can help", "our agency specializes",
    "book a call", "dm me for", "dm to book", "contact us", "get a quote", "free consultation",
    "schedule a call", "we deliver", "we're a full-service", "we are a full-service",
    "get started today", "sign up now", "limited spots", "link in bio", "white-label options for agencies",
)

JOB_SEEKER_MARKERS: tuple[str, ...] = (
    "open to work", "seeking new opportunities", "looking for new opportunities",
    "available for projects", "available for freelance", "available for work",
    "seeking a position", "seeking employment", "freelance available",
    "i'm a freelance", "i am a freelance", "hire me", "my portfolio", "resume available",
    "seeking new clients", "open for freelance", "freelance for hire",
)

MARKETPLACE_MARKERS: tuple[str, ...] = (
    "join our network", "join our platform", "apply to join", "we place", "staffing agency",
    "talent pool", "we recruit", "register as a freelancer", "sign up as a freelancer",
    "we connect brands", "submit your portfolio to join", "join our talent", "vetted network",
)

THOUGHT_LEADERSHIP_RE: tuple[re.Pattern[str], ...] = (
    re.compile(r"save this for later", re.IGNORECASE),
    re.compile(r"\bpro tip\b", re.IGNORECASE),
    re.compile(r"\btop \d+\b", re.IGNORECASE),
    re.compile(r"^\s*\d+\s+(tips|things|lessons)", re.IGNORECASE),
    re.compile(r"lessons learned", re.IGNORECASE),
    re.compile(r"here are my tips", re.IGNORECASE),
    re.compile(r"why your \w+ (is|are)", re.IGNORECASE),
)


@dataclass(slots=True)
class PrefilterVerdict:
    keep: bool
    dropped_for: str | None = None
    matched: dict[str, bool] = field(default_factory=dict)

    def summary(self) -> str:
        hits = ", ".join(k for k, v in self.matched.items() if v) or "none"
        return f"keep={self.keep} dropped_for={self.dropped_for} matched=[{hits}]"


def _contains(text: str, phrases: tuple[str, ...]) -> bool:
    return any(p in text for p in phrases)


def prefilter(text: str, *, service: str = "") -> PrefilterVerdict:
    """Decide cheaply whether a post is worth GPT-4o spend.

    Never drops a post that shows any buyer marker (the classifier owns those
    ambiguous cases). Without a buyer marker, near-certain negatives are
    dropped; anything else stays for the classifier.
    """
    low = (text or "").lower()
    matched = {
        "buyer": _contains(low, BUYER_MARKERS),
        "job_ad": _contains(low, JOB_AD_STRONG),
        "seller": _contains(low, SELLER_MARKERS),
        "job_seeker": _contains(low, JOB_SEEKER_MARKERS),
        "marketplace": _contains(low, MARKETPLACE_MARKERS),
        "thought_leadership": any(r.search(low) for r in THOUGHT_LEADERSHIP_RE),
    }
    if matched["buyer"]:
        # Ambiguous with a buyer phrase present -> classifier decides.
        return PrefilterVerdict(keep=True, matched=matched)

    for category in ("job_ad", "seller", "job_seeker", "marketplace", "thought_leadership"):
        if matched[category]:
            return PrefilterVerdict(keep=False, dropped_for=category, matched=matched)
    return PrefilterVerdict(keep=True, matched=matched)
