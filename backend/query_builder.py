"""High-signal LinkedIn search queries built from the user's service text.

Weak queries are the single biggest lever on lead quality, so this module
spends real effort phrasing buyer language per lead type instead of dumping
the service word in one generic template. Every search always runs the whole
small base set together — a single narrow phrasing is never the only query —
and the engine later dips into `pool` to diversify when it is short of N.

GENERALIZATION RULE (§0): {service} is an opaque plain string. There is no
branching on what the service *is*, and no service name is baked anywhere —
the same templates must work unmodified for a plumber, a UX designer or a
wedding photographer, purely from the value passed in.

NEGATIVE PAIRING (§6): positive-only queries return garbage, so every query
pairs the buyer phrase with quoted negative seller terms the provider can
honor (LinkedIn supports -"phrase" exclusions). These are generic sell-side
phrases, deliberately not service-specific.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from models import LeadType

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9&+'-]*")
_ARTICLE = re.compile(r"^(a|an|the)\s+(.+)$", re.IGNORECASE)
_SPACES = re.compile(r"\s+")

# Paired into EVERY query as -"..." exclusions (case-insensitive on the
# provider side; generic sell-side language, no service words).
NEGATIVE_QUERY_PHRASES: tuple[str, ...] = (
    "we offer",
    "our services",
    "book a call",
    "dm us",
    "we specialize",
    "we help",
)


_NEG_TERM = re.compile(r'\s+-\"([^\"]+)\"')


def split_query(query: str) -> tuple[str, tuple[str, ...]]:
    """Split an emitted query back into its positive phrasing and the quoted
    negative phrases. Lets providers/mocks honor the exclusions without
    treating '-"we offer"' tokens as matchable keywords."""
    positive = _NEG_TERM.sub("", query).strip()
    negatives = tuple(n.lower() for n in _NEG_TERM.findall(query))
    return positive, negatives


def _with_negatives(query: str) -> str:
    suffix = "".join(f' -"{p}"' for p in NEGATIVE_QUERY_PHRASES)
    return query + suffix


@dataclass(frozen=True, slots=True)
class QueryPlan:
    base: tuple[str, ...]  # always run together (3-5 natural phrasings)
    pool: tuple[str, ...]  # diversification candidates for later iterations


def _normalize(service: str) -> tuple[str, str, bool]:
    svc = _SPACES.sub(" ", service.strip())
    m = _ARTICLE.match(svc)
    if m:
        return svc, m.group(2).strip(), True
    return svc, svc, False


def _article_phrase(naked: str) -> str | None:
    """Return a natural 'a/an <noun phrase>' when the service is countable,
    else None (service is mass/gerund/plural and reads better bare)."""
    words = naked.split()
    if not words:
        return None
    last = words[-1].lower()
    if last.endswith("ing"):
        return None
    if last.endswith(("ss", "us", "is")):
        pass
    elif last.endswith(("s", "x", "z", "ch", "sh")):
        return None  # looks plural
    article = "an" if naked[0].lower() in "aeiou" else "a"
    return f"{article} {naked}"


def _split_service(service: str) -> tuple[str, str, str | None]:
    svc, naked, had_article = _normalize(service)
    phrase: str | None
    if had_article:
        phrase = svc
    else:
        phrase = _article_phrase(naked)
    return svc, naked, phrase


def _words(naked: str) -> list[str]:
    return _WORD.findall(naked)


# ---------------------------------------------------------------------------
# Per-lead-type phrasings (§6). {A} = article phrase ("a video editor"),
# {S} = bare service as typed, {N} = service with any leading article stripped.
# ---------------------------------------------------------------------------

def _need_freelancer(svc: str, naked: str, phrase: str | None) -> list[str]:
    A = phrase or svc
    qs = [
        f"looking for {A}",
        f"need {A}",
        f"anyone know a good {naked}" if phrase else f"recommendations for {svc}",
        f"need someone to help with {svc}",
    ]
    return [q for q in qs if q]


def _hiring_buyer(svc: str, naked: str, phrase: str | None) -> list[str]:
    A = phrase or svc
    qs = [
        f"our company needs {A}",
        f"looking for {A} budget",
        f"seeking {A} for our team",
        f"need {A} asap",
        f"we are looking for {A} for a project",
    ]
    return [q for q in qs]


def _our_agency(svc: str, naked: str, phrase: str | None) -> list[str]:
    A = phrase or svc
    qs = [
        "looking for freelancers to work with our agency",
        f"need extra {svc} help for client projects",
        f"white label {svc} looking for",
        f"looking for {svc} freelancers for client projects",
    ]
    return [q for q in qs]


_BUILDERS = {
    LeadType.NEED_FREELANCER: _need_freelancer,
    LeadType.HIRING_BUYER: _hiring_buyer,
    LeadType.OUR_AGENCY: _our_agency,
}


def build_plan(service: str, lead_type: LeadType) -> QueryPlan:
    """Build the base set + a deterministic diversification pool.

    Every emitted query pairs the buyer phrasing with the generic negative
    seller terms (see module docstring / §6). Dedupe happens on the raw
    phrasing before the shared negative suffix is appended.
    """
    svc, naked, phrase = _split_service(service)
    base = _BUILDERS[lead_type](svc, naked, phrase)
    pool = _pool(svc, naked, phrase, lead_type)
    seen: set[str] = set()
    out_b: list[str] = []
    for q in base:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out_b.append(_with_negatives(q))
    out_p: list[str] = []
    for q in pool:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out_p.append(_with_negatives(q))
    return QueryPlan(base=tuple(out_b), pool=tuple(out_p))


def _pool(svc: str, naked: str, phrase: str | None, lead_type: LeadType) -> list[str]:
    """Large deterministic diversification pool (service-agnostic).

    Sized so the exact-count engine can keep looping well past the base set —
    each phrasing is still a natural buyer sentence with the user's service
    interpolated as an opaque string. No service names are hardcoded.
    """
    A = phrase or svc  # article phrase ("a plumber") or bare ("video editing")
    N = naked          # service without a leading article
    out: list[str] = []

    if lead_type == LeadType.NEED_FREELANCER:
        # --- hiring/needing verbs ------------------------------------------
        verbs = [
            "looking for", "searching for", "in need of", "on the lookout for",
            "trying to find", "looking to find", "hunting for", "want to hire",
            "looking to hire", "need to hire", "looking for recommendations for",
            "in the market for", "on the hunt for",
        ]
        out += [f"{v} {A}" for v in verbs]
        # --- urgency / timing ----------------------------------------------
        for u in ("urgent", "urgently", "asap", "this week", "for next week", "soon"):
            out.append(f"need {A} {u}")
        out.append(f"looking for {A} paid")
        out.append(f"need {A} with a budget")
        # --- referral language ----------------------------------------------
        out.append(f"anyone know a good {N}")
        out.append(f"does anyone know a good {N}")
        out += [f"anyone recommend {A}", f"can anyone recommend {A}", f"who can recommend {A}"]
        out.append(f"recommendations for {A}")
        out.append(f"recommendations needed for {A}")
        # --- help/handle forms ----------------------------------------------
        out += [
            f"need someone to help with {svc}",
            f"need help with {svc}",
            f"looking for someone to help with {svc}",
            f"need someone to handle {svc}",
            f"looking for someone to handle {svc}",
        ]
        # --- short/classified forms ----------------------------------------
        out += [
            f"{N} needed",
            f"freelance {N} needed",
            f"need a freelance {N}",
            f"looking for a freelance {N}",
            "can anyone point me to someone good",
            f"looking to bring on {A}",
            f"my team needs {A}",
            f"we are looking for {A} for some work",
            f"looking for a freelance {N} for a one-off job",
            f"need an independent {N} for a short project",
            f"anyone know a freelance {N} who can help",
        ]
        return out

    if lead_type == LeadType.HIRING_BUYER:
        entities = ["our company", "our team", "my company", "our startup", "our business", "our firm", "our studio"]
        out += [f"{e} needs {A}" for e in entities]
        out += [f"{e} is looking for {A}" for e in entities]
        out += [f"{e} is hiring {A}" for e in entities]
        # --- scoped requests ------------------------------------------------
        scopes = [
            "for a project", "for our team", "for our company", "for a campaign",
            "on contract", "as a freelancer", "freelance", "remote", "to start this week",
        ]
        out += [f"looking for {A} {s}" for s in scopes]
        out += [f"seeking {A} {s}" for s in scopes]
        # --- budget / urgency ----------------------------------------------
        out += [
            f"looking for {A} budget approved",
            f"looking for {A} with budget",
            f"need {A} within budget",
            f"we need {A} asap",
            f"need {A} urgently for a client",
            f"we are looking for {A} for a project",
            f"we need {A} for an upcoming project",
            f"{svc} needed for our company",
            f"{svc} needed for our team",
            f"hiring {A} on a project basis",
            f"hiring a freelance {N}",
            f"our company is looking to hire a freelance {N}",
            f"we are hiring a freelance {N} for a paid project",
            f"looking to hire {A} for our team",
            f"need {A} for our company, budget approved",
        ]
        return out

    # OUR_AGENCY
    out += [
        "looking for freelancers to work with our agency",
        f"looking for {svc} freelancers to work with our agency",
        f"{svc} freelancers needed for client projects",
        f"need extra {svc} help for client projects",
        f"our agency needs extra {svc} help",
        f"looking for extra {svc} help for our clients",
        f"white label {svc} partner wanted",
        f"looking for a white label {svc} partner",
        f"white label {svc} freelancers looking for",
        f"outsource {svc} for client projects",
        f"need {svc} contractors for client projects",
        f"looking for {svc} contractors for client work",
        f"our agency is looking for freelance {svc} for client projects",
        f"need {svc} subcontractors for client projects",
        f"{svc} overflow help for our agency",
        f"overflow {svc} work for our agency",
        f"looking for vetted {svc} freelancers",
        f"hiring freelance {svc} for client projects",
        f"bring on {svc} freelancers for client work",
        f"looking to bring on {svc} freelancers",
        f"partner with {svc} freelancers for client projects",
        f"need {svc} help for client work this quarter",
        f"looking for freelance help for client projects",
        "need a freelance bench for client work",
        f"any good {svc} freelancers for agency work",
    ]
    return out


def next_queries(service: str, lead_type: LeadType, iteration: int) -> list[str]:
    """Queries to run in engine `iteration` (0 = base set only)."""
    plan = build_plan(service, lead_type)
    if iteration <= 0:
        return list(plan.base)
    # Later iterations add a widening slice of the pool while always keeping
    # the two strongest base phrasings (protection against provider flakiness).
    window = 3 + iteration * 2
    start = (iteration - 1) * 2
    extra = list(plan.pool[start : start + window])
    if not extra:
        return []
    return list(plan.base[:2]) + extra
