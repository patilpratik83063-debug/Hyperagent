"""Canonical domain model: lead types, time windows, and API schemas.

Lead-type taxonomy is the single source of truth for the whole system.
Everything else (query builder, classifier prompt, scoring, schema.sql)
must stay consistent with it.
"""
from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LeadType(str, Enum):
    """Buyer intents that are requestable. Everything else is rejected."""

    NEED_FREELANCER = "need_freelancer"  # individual/founder wants an independent freelancer
    HIRING_BUYER = "hiring_buyer"  # a business actively hiring for a project/role
    OUR_AGENCY = "our_agency"  # an agency wants to bring in outside freelance help


LEAD_TYPE_LABELS: dict[LeadType, str] = {
    LeadType.NEED_FREELANCER: "Need Freelancer",
    LeadType.HIRING_BUYER: "Hiring Buyer",
    LeadType.OUR_AGENCY: "Our Agency",
}

# Sent the classifier may return when a candidate is not a qualified buyer.
IRRELEVANT = "irrelevant"
CLASSIFICATION_TYPES = tuple(t.value for t in LeadType) + (IRRELEVANT,)


class TimeWindow(str, Enum):
    """Freshness window for a search / re-filter of saved leads."""

    HOURS_24 = "24h"
    DAYS_7 = "7d"
    DAYS_14 = "14d"
    DAYS_28 = "28d"

    def days(self) -> int:
        return {"24h": 1, "7d": 7, "14d": 14, "28d": 28}[self.value]

    def label(self) -> str:
        return {"24h": "Last 24h", "7d": "Last 7 days", "14d": "Last 14 days", "28d": "Last 28 days"}[self.value]

    def cutoff(self, now: datetime | None = None) -> datetime:
        """Discovery-time cutoff.

        Computed fresh on every call from *now* — never stored/hardcoded.
        The cutoff is rounded down to the start of the UTC day `window` days
        ago, i.e. the widest bucket that guarantees at least one full day of
        posts and never less than the requested freshness.
        """
        now = now or datetime.now(UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        start = (now - timedelta(days=self.days())).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
        return start

    def after_date(self, now: datetime | None = None) -> date:
        """Concrete `after:YYYY-MM-DD` used by discovery providers."""
        return self.cutoff(now).date()

    def after_iso(self, now: datetime | None = None) -> str:
        return self.cutoff(now).date().isoformat()


class IntentStrength(str, Enum):
    """Direction-of-intent ladder used for ranking and the acceptance gate."""

    EXPLICIT = "explicit"  # clear procurement ask, author is the buyer/owner
    ACTIVE_SEARCH = "active_search"  # author is actively sourcing right now
    RECOMMENDATION = "recommendation"  # author asks for recommendations/referrals
    PROBLEM_AWARENESS = "problem_awareness"  # problem stated, no ask yet
    RESEARCH = "research"  # exploring the market
    NONE = "none"

    def rank(self) -> int:
        return {
            IntentStrength.EXPLICIT: 100,
            IntentStrength.ACTIVE_SEARCH: 85,
            IntentStrength.RECOMMENDATION: 70,
            IntentStrength.PROBLEM_AWARENESS: 50,
            IntentStrength.RESEARCH: 30,
            IntentStrength.NONE: 0,
        }[self]


# ---------------------------------------------------------------------------
# Classifier structured output (OpenAI JSON-schema / strict mode target)
# ---------------------------------------------------------------------------

class LeadClassification(BaseModel):
    """Structured classification returned by GPT-4o for one candidate post."""

    model_config = ConfigDict(extra="forbid")

    lead_type: Literal["need_freelancer", "hiring_buyer", "our_agency", "irrelevant"]
    intent_strength: Literal[
        "explicit", "active_search", "recommendation", "problem_awareness", "research", "none"
    ]
    is_buying_sourcing: bool = Field(description="Author needs the service for their own project/company")
    is_selling_offering: bool = Field(description="Author provides/sells the service themselves")
    is_job_seek: bool = Field(description="Author is a freelancer/employee seeking work for themselves")
    service_match_score: float = Field(ge=0, le=100)
    commercial_intent_score: float = Field(ge=0, le=100)
    decision_maker_signal: bool = Field(description="Author appears to be a decision-maker/budget-owner")
    evidence_strength: float = Field(ge=0, le=100)
    overall_quality_score: float = Field(ge=0, le=100)
    is_qualified: bool = Field(description="Model's own verdict; engine still applies hard gates")
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(description="Short verbatim quote/paraphrase of the deciding line")
    reason: str = Field(description="Brief justification including which trap (if any) was ruled out")


# ---------------------------------------------------------------------------
# API schemas
# ---------------------------------------------------------------------------

_LEAD_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: str = Field(min_length=3, max_length=300, description="Free-text service description, e.g. 'a plumber', 'UX design', 'wedding photography'")
    country: str = Field(default="", max_length=120)
    lead_type: LeadType
    time_window: TimeWindow = TimeWindow.DAYS_7
    leads_needed: int = Field(default=25, ge=1, le=500)

    @field_validator("service")
    @classmethod
    def _service_is_meaningful(cls, v: str) -> str:
        v = re.sub(r"\s+", " ", v.strip())
        if len(v) < 3:
            raise ValueError("service must be at least 3 characters")
        return v

    @field_validator("country")
    @classmethod
    def _country_strip(cls, v: str) -> str:
        return v.strip() if v else ""


class LeadPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["new", "contacted", "replied", "not_a_fit"] | None = None
    notes: str | None = Field(default=None, max_length=5000)


class SearchCreateResponse(BaseModel):
    search_id: str
    status: str = "queued"


class UsageOut(BaseModel):
    used: int
    limit: int
    remaining: int
    resets_in_seconds: int | None = None


class LeadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    search_id: str | None = None
    lead_type: str | None = None
    time_window: str | None = None
    post_url: str
    author_name: str | None = None
    author_profile_url: str | None = None
    post_text: str | None = None
    post_date: date | None = None
    overall_quality_score: float | None = None
    service_match_score: float | None = None
    intent_strength: str | None = None
    status: str = "new"
    notes: str | None = None
    created_at: datetime | None = None


def now_utc() -> datetime:
    return datetime.now(UTC)
