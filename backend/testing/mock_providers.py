"""Offline providers for demo mode and tests (no external keys needed).

The corpus deliberately mixes genuine buyers of every lead type with the §2
trap categories (sellers, job seekers, talent marketplaces, agency
self-promotion, thought leadership, job ads), so engine behavior can be
verified without spending real discovery/AI calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from discovery.base import RawPost, SearchBatchResult, canonical_post_url
from models import IntentStrength, LeadClassification, LeadType
from query_builder import split_query

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

_STOP = {
    "a", "an", "the", "for", "our", "to", "we", "are", "is", "of", "in", "on", "and", "or",
    "anyone", "know", "good", "someone", "need", "needs", "looking", "help", "with", "asap",
    "budget", "project", "client", "freelance", "freelancers", "work", "white", "label", "company",
}


def _service_words(service: str) -> list[str]:
    return [w.lower().rstrip("s") for w in service.split() if w.lower() not in _STOP and len(w) > 2]


@dataclass(slots=True)
class FakePost:
    url: str
    text: str
    author: str
    author_url: str
    days_ago: float
    lead_type: LeadType | str
    intent: IntentStrength
    service_match: float
    commercial: float
    decision_maker: bool
    evidence: str
    reason: str
    include_in_generic: bool = True  # show for arbitrary service text too


def _now(days_ago: float) -> datetime:
    return datetime.now(UTC) - timedelta(days=days_ago)


_C = 0


def _mk(
    text: str, lead_type: LeadType | str, intent: IntentStrength, *, days: float, match: float = 90.0,
    commercial: float = 80.0, decision: bool = True, evidence: str = "", reason: str = "",
    generic: bool = True,
) -> FakePost:
    global _C
    _C += 1
    return FakePost(
        url=f"https://www.linkedin.com/posts/mock-{_C:04d}-abcdef",
        text=text,
        author=f"Mock Author {_C}",
        author_url=f"https://www.linkedin.com/in/mock-author-{_C}",
        days_ago=days,
        lead_type=lead_type,
        intent=intent,
        service_match=match,
        commercial=commercial,
        decision_maker=decision,
        evidence=evidence or text[:140],
        reason=reason or f"corpus post {_C}",
        include_in_generic=generic,
    )


# (text contains "video editor" / "video editing" / "animation" style tokens — the
# generic corpus is service-agnostic: engine tests use service "video editor", and
# posts below that mention "video" are boosted by include_in_generic scoring below.)
CORPUS: list[FakePost] = [
    # --- genuine: need freelancer -------------------------------------------------
    _mk("We just wrapped a launch and are looking for a freelance video editor to help cut our case studies. DM if you know someone great.",
        LeadType.NEED_FREELANCER, IntentStrength.EXPLICIT, days=0.2, evidence="looking for a freelance video editor"),
    _mk("Anyone know a good video editor for a short promo? Budget is ready, need it in two weeks.",
        LeadType.NEED_FREELANCER, IntentStrength.ACTIVE_SEARCH, days=1.1, commercial=90),
    _mk("I need a video editor who can handle 3 short clips a week for my YouTube channel. Send portfolios.",
        LeadType.NEED_FREELANCER, IntentStrength.EXPLICIT, days=3.0),
    _mk("Does anyone have recommendations for a video editor for wedding films? We shoot in Austin.",
        LeadType.NEED_FREELANCER, IntentStrength.RECOMMENDATION, days=6.0, evidence="recommendations for a video editor"),
    _mk("Need someone to handle video editing for our podcast episodes going forward. Paid, monthly retainer.",
        LeadType.NEED_FREELANCER, IntentStrength.ACTIVE_SEARCH, days=9.0),
    _mk("Searching for a video editor to fix 20 legacy training videos, ~40h of work. Happy to pay market rate.",
        LeadType.NEED_FREELANCER, IntentStrength.ACTIVE_SEARCH, days=12.0),
    _mk("Looking for a freelance motion designer / video editor for our internal explainers. Portfolio links welcome.",
        LeadType.NEED_FREELANCER, IntentStrength.ACTIVE_SEARCH, days=0.5, match=80),
    _mk("Our founder needs a video editor for a keynote speech edit, one-off. Anyone free this week?",
        LeadType.NEED_FREELANCER, IntentStrength.EXPLICIT, days=2.0),
    # --- genuine: hiring buyer -----------------------------------------------------
    _mk("Our company needs a video editing agency or freelancer for a rebrand campaign. We have an approved budget and timeline.",
        LeadType.HIRING_BUYER, IntentStrength.EXPLICIT, days=0.4, commercial=95,
        evidence="Our company needs ... approved budget"),
    _mk("Looking for a video editor for our marketing team, project-based with possible retainer. Budget approved.",
        LeadType.HIRING_BUYER, IntentStrength.ACTIVE_SEARCH, days=1.5),
    _mk("Seeking a video editor for our SaaS launch videos. We're a 40-person startup, decision this week.",
        LeadType.HIRING_BUYER, IntentStrength.ACTIVE_SEARCH, days=4.0, decision=True),
    _mk("Need a video editor asap for an investor deck trailer — deadline Friday, contractor fine.",
        LeadType.HIRING_BUYER, IntentStrength.EXPLICIT, days=0.1, commercial=95),
    _mk("We are hiring a freelance video editor for a 6-week campaign, not a full-time role. Rate negotiable within budget.",
        LeadType.HIRING_BUYER, IntentStrength.ACTIVE_SEARCH, days=5.0, match=85),
    # --- genuine: our agency -------------------------------------------------------
    _mk("We're a video production agency looking for freelancers to work with our agency on overflow client projects.",
        LeadType.OUR_AGENCY, IntentStrength.EXPLICIT, days=0.8, evidence="looking for freelancers to work with our agency"),
    _mk("Our agency needs extra video editing help for client projects this quarter. White-label collaboration welcome.",
        LeadType.OUR_AGENCY, IntentStrength.ACTIVE_SEARCH, days=2.5, commercial=90),
    _mk("Looking for vetted freelance video editors to join our delivery bench for client work. Paid per project.",
        LeadType.OUR_AGENCY, IntentStrength.ACTIVE_SEARCH, days=7.0),
    # --- traps: seller / self-promotion --------------------------------------------
    _mk("We offer video editing services — book a call for a free consultation and a quote within 24h.",
        "irrelevant", IntentStrength.NONE, days=0.3, match=10, commercial=10,
        reason="seller/offering: author provides the service"),
    _mk("DM me for video editing! I deliver 48h turnaround, white-label options for agencies.",
        "irrelevant", IntentStrength.NONE, days=1.0, match=15, reason="seller/offering: author provides the service"),
    _mk("Our agency specializes in video editing for B2B. We can help your brand stand out — get in touch.",
        "irrelevant", IntentStrength.NONE, days=0.6, match=12, reason="agency self-promotion, not our_agency"),
    _mk("Need video editing? We are a full-service production house with 15 years of experience. Sign up for a call.",
        "irrelevant", IntentStrength.NONE, days=2.2, match=10, reason="seller/offering"),
    # --- traps: job seekers ----------------------------------------------------------
    _mk("I'm a freelance video editor available for projects — open to work, portfolio in comments.",
        "irrelevant", IntentStrength.NONE, days=1.2, match=20, reason="job seeker: wants employment for themselves"),
    _mk("Freelance video editor seeking new clients and opportunities for 2025. Links below.",
        "irrelevant", IntentStrength.NONE, days=3.5, match=18, reason="job seeker"),
    # --- traps: talent marketplace / recruiting-seller ------------------------------
    _mk("Our platform connects brands with vetted video editors. Join our talent network today — free for freelancers.",
        "irrelevant", IntentStrength.NONE, days=0.9, reason="talent marketplace recruiting freelancers to place elsewhere"),
    _mk("We are a staffing agency placing video editors at enterprise clients. Submit your portfolio to join our pool.",
        "irrelevant", IntentStrength.NONE, days=4.5, reason="recruiting-seller"),
    # --- traps: thought leadership ---------------------------------------------------
    _mk("5 video editing tips that will double your retention. Save this for later 🚀",
        "irrelevant", IntentStrength.NONE, days=0.5, match=5, reason="thought leadership: generic tips, no procurement"),
    _mk("In my experience, good video editing is 80% storytelling. Here is my take on the craft.",
        "irrelevant", IntentStrength.NONE, days=6.5, match=5, reason="thought leadership"),
    # --- traps: job ads ----------------------------------------------------------------
    _mk("We are hiring a full-time video editor. Apply now at careers.example.com — vacancy #4421.",
        "irrelevant", IntentStrength.NONE, days=1.4, match=20, reason="job ad / vacancy"),
    _mk("Full-time position open: in-house video editor, benefits included. Apply before Friday.",
        "irrelevant", IntentStrength.NONE, days=8.0, match=15, reason="job ad"),
]


class MockDiscoveryClient:
    """Returns corpus posts whose text overlaps the query, in rotating slices.

    Simulates paging: each successive call for the same query returns the
    *next* unseen slice of matches, so the engine's iteration logic
    (keep scanning until N found or results run dry) is exercised for real.
    """

    name = "mock"

    def __init__(self, corpus: list[FakePost] | None = None) -> None:
        self.corpus = corpus if corpus is not None else CORPUS
        self._cursor: dict[str, int] = {}
        self.calls: int = 0

    @property
    def config_errors(self) -> list[str]:
        return []

    def search_posts(self, queries, since, *, results_per_query=25) -> SearchBatchResult:  # type: ignore[no-untyped-def]
        posts: list[RawPost] = []
        self.calls += 1
        now = datetime.now(UTC)
        for query in queries:
            positive, negatives = split_query(query)
            words = _service_words(positive)
            matches = [
                p
                for p in self.corpus
                if any(w in p.text.lower() for w in words)
                and not any(n in p.text.lower() for n in negatives)  # quoted -"..." exclusions
                and (now - timedelta(days=p.days_ago)) >= since
            ]
            matches.sort(key=lambda p: p.days_ago)
            start = self._cursor.get(query, 0)
            chunk = matches[start : start + max(results_per_query, 1)]
            self._cursor[query] = start + len(matches)
            for p in chunk:
                posts.append(
                    RawPost(
                        post_url=p.url,
                        text=p.text,
                        author_name=p.author,
                        author_profile_url=p.author_url,
                        posted_at=_now(p.days_ago),
                        query_used=query,
                        provider=self.name,
                    )
                )
        seen: set[str] = set()
        unique: list[RawPost] = []
        for p in posts:
            key = canonical_post_url(p.post_url)
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return SearchBatchResult(posts=unique, queries_used=list(queries))


class MockClassifier:
    """Deterministic stand-in for the GPT-4o classifier (offline only)."""

    def __init__(self, corpus: list[FakePost] | None = None) -> None:
        self.corpus = corpus if corpus is not None else CORPUS
        self._by_url = {p.url: p for p in self.corpus}

    @property
    def config_errors(self) -> list[str]:
        return []

    def classify_batch(self, candidates, *, service, lead_type, country="", max_concurrency=8):  # type: ignore[no-untyped-def]
        results: list[LeadClassification | None] = []
        for cand in candidates:
            post = self._by_url.get(cand.post_url)
            if post is None:
                results.append(None)
                continue
            lt = post.lead_type if isinstance(post.lead_type, str) else post.lead_type.value
            results.append(
                LeadClassification(
                    lead_type=lt,
                    intent_strength=post.intent.value,
                    is_buying_sourcing=lt in (t.value for t in LeadType),
                    is_selling_offering=lt == "irrelevant" and "offer" in post.text.lower(),
                    is_job_seek=lt == "irrelevant" and post.reason.startswith("job seeker"),
                    service_match_score=post.service_match,
                    commercial_intent_score=post.commercial,
                    decision_maker_signal=post.decision_maker,
                    evidence_strength=85.0,
                    overall_quality_score=(post.service_match + post.commercial) / 2,
                    is_qualified=lt in (t.value for t in LeadType),
                    confidence=0.99,
                    evidence=post.evidence,
                    reason=post.reason,
                )
            )
        return results


def build_service_corpus(
    terms: tuple[str, ...],
    *,
    location: str = "",
    include_traps: bool = True,
) -> list[FakePost]:
    """Deterministic offline corpus for ANY service/location.

    Used by the generalization tests (§0): the same pipeline must work for a
    plumber, a UX designer or a wedding photographer purely from the values
    passed in. `terms` are the service words that must appear in post text
    (matching what the engine's queries will carry), `location` is embedded
    into a couple of posts so a country/location signal can be verified.
    """
    svc = " ".join(terms)
    svc_a = f"a {svc}" if not svc.lower().startswith(("a ", "an ", "the ")) else svc
    svc_s = svc + "s" if not svc.endswith("s") else svc
    loc = location or ""
    in_loc = f" in {loc}" if loc else ""
    at_loc = f", {loc}" if loc else ""

    out: list[FakePost] = []
    last = {"n": 0}

    def add(text: str, lead_type: LeadType | str, intent: IntentStrength, *,
            days: float, match: float = 88.0, commercial: float = 82.0,
            decision: bool = True, reason: str = "", evidence: str = "") -> None:
        last["n"] += 1
        out.append(FakePost(
            url=f"https://www.linkedin.com/posts/gen-{last['n']:04d}-{svc.replace(' ', '-')}",
            text=text,
            author=f"Generic Author {last['n']}",
            author_url=f"https://www.linkedin.com/in/generic-{last['n']}",
            days_ago=days,
            lead_type=lead_type,
            intent=intent,
            service_match=match,
            commercial=commercial,
            decision_maker=decision,
            evidence=evidence or text[:140],
            reason=reason or f"generic corpus post {last['n']}",
            include_in_generic=True,
        ))

    # --- genuine buyers ----------------------------------------------------
    add(f"We are looking for {svc_a} to handle some urgent work for us{in_loc}. Can anyone recommend someone?",
        LeadType.NEED_FREELANCER, IntentStrength.ACTIVE_SEARCH, days=0.4,
        reason="genuine need_freelancer buyer")
    add(f"Need {svc_a} for a one-off job next week. Budget is ready, happy to pay market rate.",
        LeadType.NEED_FREELANCER, IntentStrength.EXPLICIT, days=2.2,
        commercial=92, reason="genuine need_freelancer buyer")
    add(f"Anyone know a good {svc}? We have a small project{at_loc} that needs doing.",
        LeadType.NEED_FREELANCER, IntentStrength.RECOMMENDATION, days=6.0,
        reason="genuine need_freelancer buyer")
    add(f"Our company needs a {svc} for a project{in_loc}. Approved budget and a tight timeline.",
        LeadType.HIRING_BUYER, IntentStrength.EXPLICIT, days=1.0,
        commercial=95, decision=True, reason="genuine hiring_buyer buyer")
    add(f"Seeking a {svc} for our team — project-based with possible retainer. Budget approved.",
        LeadType.HIRING_BUYER, IntentStrength.ACTIVE_SEARCH, days=3.5,
        reason="genuine hiring_buyer buyer")
    add(f"We're a local agency{at_loc} looking for freelance {svc_s} to work with our agency on client projects.",
        LeadType.OUR_AGENCY, IntentStrength.EXPLICIT, days=1.8,
        reason="genuine our_agency buyer")

    if include_traps:
        add(f"We offer professional {svc} services — book a call for a free quote.",
            "irrelevant", IntentStrength.NONE, days=1.3, match=10, commercial=10,
            reason="seller/offering: author provides the service")
        add(f"Our agency specializes in {svc} for businesses. We can help your brand — get in touch.",
            "irrelevant", IntentStrength.NONE, days=2.0, match=12,
            reason="agency self-promotion, not our_agency")
        add(f"I'm a freelance {svc} open to work — available for projects immediately.",
            "irrelevant", IntentStrength.NONE, days=0.9, match=15,
            reason="job seeker: wants employment for themselves")
        add(f"Join our platform to book vetted {svc_s} for your needs. Sign up today.",
            "irrelevant", IntentStrength.NONE, days=1.6, match=8,
            reason="talent marketplace recruiting freelancers to place elsewhere")
        add(f"We are hiring a full-time {svc} for our office. Apply now — vacancy #12.",
            "irrelevant", IntentStrength.NONE, days=5.2, match=18,
            reason="job ad / vacancy")
        add(f"5 {svc} tips that will save you thousands. Save this for later.",
            "irrelevant", IntentStrength.NONE, days=4.8, match=5,
            reason="thought leadership: generic tips, no procurement")
    return out
