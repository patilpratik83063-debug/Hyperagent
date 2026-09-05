"""GPT-4o structured lead classification — fail-closed by design (§7).

Direction-of-intent is the core question: WHO NEEDS vs WHO OFFERS. Every
seller trap from §2 is spelled out in the system prompt with examples.

Fail-closed rules:
  * no API key / model unavailable      -> candidates are dropped, never guessed
  * timeout / API error after retries    -> candidate dropped
  * response fails Pydantic validation   -> candidate dropped (retried once)
A dropped candidate simply does not become a lead; one bad response can never
fail or corrupt the whole batch (per-candidate isolation, bounded concurrency).
"""
from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from models import LeadClassification
from pydantic import ValidationError

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the lead-qualification classifier for a service marketplace. Your only job is to
decide whether a LinkedIn post is a GENUINE BUYER of the service the user sells — a real
procurement intent from someone who needs the service — or noise that must be rejected.

Ask ONE question about every post: WHO NEEDS and WHO OFFERS?

Answer as one of four types:
- need_freelancer: an individual or founder wants an independent freelancer for themselves.
- hiring_buyer: a BUSINESS is actively hiring for a project/role (company context, budget, team).
- our_agency: an AGENCY wants to bring in outside freelance help (they source, they do not sell).
- irrelevant: everything else — see the reject categories below.

REJECT CATEGORIES — recognize and reject each, even in disguise:
1. Seller / offering: the author provides the service themselves. Signals: "we offer...",
   "DM me for...", "book a call", "free consultation", "white-label/OEM pitch", a supplier
   pitching a lead-list TO agencies, "we can help your brand", "limited spots".
2. Job seeker: the author wants employment or clients for THEMSELVES. Signals: "open to work",
   "I'm a freelance X available for projects", "seeking new clients", "portfolio in comments".
3. Talent marketplace / recruiting-seller: a platform or staffing agency recruiting freelancers
   to place at SOMEONE ELSE'S clients. Signals: "join our talent network", "we place X at",
   "staffing agency", "submit your portfolio to join our pool". This is NOT a buyer, even when
   they say they need freelancers — they recruit freelancers as inventory, not for their own project.
4. Agency self-promotion: "our agency can help", "we specialize in X", "full-service agency".
   The word "agency" does NOT make it our_agency — offering help is SELLING. Only an agency that
   is sourcing help for its own client work is our_agency.
5. Thought leadership: generic tips/advice/opinion with no procurement action ("5 tips...",
   "in my experience...", "why your X is broken"). No ask = no lead.
6. Job ads for employees: "full-time position", "vacancy", "apply now", "benefits". A post hiring
   a freelance/contract worker for a defined project CAN be a buyer — judge on project vs employment.

DISTINCTION RULES (precision over recall — a false positive is costly):
- "is_buying_sourcing" = the author needs the service for their OWN project/company/clients.
- Only set is_selling_offering / is_job_seek for sell-side intent; a buyer may also be a freelancer
  by profession as long as THIS post is them buying.
- our_agency requires the author to be an agency SEEKING outside help ("looking for freelancers to
  work with our agency", "extra {service} for client projects", "white-label partner wanted").
  "We offer white-label X" is a SELLER.
- A single post can mix signals ("need a {service} — DM me if you know someone" is a BUYER asking
  for referrals; "DM me for {service}" is a SELLER). Judge the main intent.

DOMAIN-GENERAL REASONING (§0): apply the direction-of-intent test below to whatever service the
user sells — plumbing in Nairobi, UX design in Toronto, wedding photography in Mumbai, anything.
There are no per-service rules; these examples teach the PATTERN, not an allow-list. The four
canonical pairs (paraphrased, for an imaginary production service) are:

1. "We're launching a product next month and need someone to shoot our ad — any recommendations for
   a production house?"  ->  is_qualified: true, lead_type: hiring_buyer (real client, genuine ask).
2. "Looking for brands to partner with for their ad films — we bring creative + production in-house."
   ->  is_qualified: false, is_selling_offering: true (direction reversed — they hunt clients).
3. "We help brands create stunning ad films that convert. DM to discuss your next project."
   ->  is_qualified: false, is_selling_offering: true (classic seller pitch).
4. "I'm a video editor/filmmaker looking for freelance projects, available immediately."
   ->  is_qualified: false, is_job_seek: true (job-seeker, not a buyer).

Same pattern for any {service}: the asker must need the work done, not offer to do it, not be a
freelancer hunting for their own gig, and not be a recruiter stocking talent for other people.

SCORING RUBRIC (0-100):
- service_match_score: how directly the post is about the service the user sells (exact service
  mentioned = 90+, adjacent craft = 60-80, unrelated = <40).
- commercial_intent_score: money signals — budget, paid, rate, timeline, deadline, retainer,
  "decision this week", scale of work.
- decision_maker_signal: true only if the author appears to own the need/budget (founder, owner,
  marketing lead, agency principal). Referrals "anyone know a good X" are still strong leads:
  set true when they ask on behalf of a need they own.
- evidence_strength: how explicit the deciding lines are (explicit ask + budget = 90+; vague =
  40-60; nothing concrete = <40).
- overall_quality_score: expected lead value 0-100 with a precision bias — overestimate only when
  evidence is strong.

Also return:
- intent_strength: explicit (clear procurement ask) | active_search (sourcing right now) |
  recommendation (asking for referrals/recommendations) | problem_awareness (problem stated, no ask) |
  research (exploring) | none.
- is_qualified: your holistic verdict. Be strict: only true when this is a real, current buyer.
- confidence: 0-1 in your verdict.
- evidence: a SHORT near-verbatim quote of the deciding line(s) from the post.
- reason: 1-2 sentences; name the category/type and, if you rejected, which trap it was.

Never invent facts not in the post. Never output anything but the JSON object."""


class ClassifierConfigError(RuntimeError):
    pass


# Hand-rolled JSON schema (strict mode requires every property + no defaults).
_TYPE_DEFINITIONS: dict[str, tuple[str, str]] = {
    "need_freelancer": (
        "Need Freelancer",
        "an INDIVIDUAL or founder/owner who wants to hire an independent freelancer for their own "
        "need — the author speaks personally or for their own small operation, not as a company "
        "recruiting a role and not as an agency.",
    ),
    "hiring_buyer": (
        "Hiring Buyer",
        "a BUSINESS/company/team actively hiring a freelancer, contractor or specialist for a project "
        "or role — company context, team, budget, urgency, timeline, or someone hiring on behalf of a "
        "business.",
    ),
    "our_agency": (
        "Our Agency",
        "an AGENCY (design/production/marketing/etc.) that wants to bring in OUTSIDE freelance help "
        "for its own client work — the author is the agency sourcing freelancers, never an agency "
        "offering its own services.",
    ),
}
_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "lead_type", "intent_strength", "is_buying_sourcing", "is_selling_offering", "is_job_seek",
        "service_match_score", "commercial_intent_score", "decision_maker_signal", "evidence_strength",
        "overall_quality_score", "is_qualified", "confidence", "evidence", "reason",
    ],
    "properties": {
        "lead_type": {"type": "string", "enum": ["need_freelancer", "hiring_buyer", "our_agency", "irrelevant"]},
        "intent_strength": {
            "type": "string",
            "enum": ["explicit", "active_search", "recommendation", "problem_awareness", "research", "none"],
        },
        "is_buying_sourcing": {"type": "boolean"},
        "is_selling_offering": {"type": "boolean"},
        "is_job_seek": {"type": "boolean"},
        "service_match_score": {"type": "number"},
        "commercial_intent_score": {"type": "number"},
        "decision_maker_signal": {"type": "boolean"},
        "evidence_strength": {"type": "number"},
        "overall_quality_score": {"type": "number"},
        "is_qualified": {"type": "boolean"},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
        "reason": {"type": "string"},
    },
}


def parse_and_validate(payload: str | dict[str, Any]) -> LeadClassification | None:
    """Parse a classifier response and validate against the Pydantic model.

    Returns None on ANY failure (fail-closed)."""
    try:
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload
        if not isinstance(data, dict):
            return None
        return LeadClassification(**data)
    except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
        log.warning("Classifier response failed validation (dropped): %s", exc)
        return None


def _user_prompt(post: dict[str, Any], service: str, country: str, requested_type: str) -> str:
    header = f"The user sells: {service}." + (f" Target location: {country}." if country else "")
    label, definition = _TYPE_DEFINITIONS.get(requested_type, (requested_type, ""))
    scope = (
        f"THIS SEARCH ACCEPTS ONLY ONE lead type: **{label}** ({requested_type}). "
        f"Definition: {definition} "
        "Only posts matching THAT exact buyer situation may be is_qualified=true. "
        "A genuine buyer whose situation belongs to a DIFFERENT bucket (or any seller, job seeker, talent "
        "marketplace, agency self-promotion or thought-leadership post) must have is_qualified=false — label its "
        "real lead_type truthfully (or irrelevant for non-buyers) and explain the type/kind mismatch in `reason`."
    )
    return f"""{header}
{scope}
Classify this LinkedIn post:

{json.dumps(post, ensure_ascii=False)}"""


class GptClassifier:
    """Structured-output GPT-4o classifier with per-candidate failure isolation."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self._client = None
        self._client_lock = threading.Lock()
        if api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)
            log.info("Classifier ready with model %s", model)

    @property
    def config_errors(self) -> list[str]:
        if self._client is None:
            return ["OPENAI_API_KEY is not set — classification is fail-closed and no candidates will be accepted"]
        return []

    def classify_batch(
        self,
        candidates,
        *,
        service: str,
        lead_type,
        country: str = "",
        max_concurrency: int = 8,
    ) -> list[LeadClassification | None]:
        """Classify many posts concurrently. Aligned with `candidates`; a
        post whose call failed/returned invalid JSON is None (dropped)."""
        if self._client is None:
            raise ClassifierConfigError("OpenAI client is not configured (OPENAI_API_KEY missing)")
        cands = list(candidates)
        if not cands:
            return []
        workers = max(1, min(max_concurrency, len(cands)))
        results: list[LeadClassification | None] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for c in cands:
                futures.append(
                    pool.submit(
                        self._classify_one,
                        c,
                        service=service,
                        requested_type=getattr(lead_type, "value", str(lead_type)),
                        country=country,
                    )
                )
            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception:  # noqa: BLE001 - never let one post kill the batch
                    log.exception("Classifier worker crashed for a candidate (dropped, fail-closed)")
                    results.append(None)
        return results

    def _classify_one(self, post, *, service: str, requested_type: str, country: str) -> LeadClassification | None:
        payload = {
            "url": post.post_url,
            "author": post.author_name,
            "author_profile_url": post.author_profile_url,
            "posted_at": post.posted_at.isoformat() if post.posted_at else None,
            "text": post.text,
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(payload, service, country, requested_type)},
        ]
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "lead_classification",
                            "strict": True,
                            "schema": _CLASSIFICATION_SCHEMA,
                        },
                    },
                )
                content = resp.choices[0].message.content
                parsed = parse_and_validate(content or "")
                if parsed is None and attempt < self.max_retries:
                    continue  # retry once on validation failure
                return parsed
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("Classifier call failed for %s (attempt %d/%d): %s",
                            post.post_url, attempt + 1, self.max_retries + 1, exc)
                time.sleep(1.5 * (attempt + 1))
        log.error("Classifier permanently failed for %s (dropped, fail-closed): %s", post.post_url, last_err)
        return None
