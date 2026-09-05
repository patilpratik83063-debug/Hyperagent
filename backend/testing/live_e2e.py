"""LIVE end-to-end smoke: real Supabase + real GPT-4o + full engine.

Discovery is the offline corpus (real Google-SERP discovery needs a
SERPER_API_KEY); everything else — engine, gates, GPT-4o structured
classification, Supabase CRUD — runs against the real services configured in
backend/.env. Run:  python testing/live_e2e.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("live_e2e")

from config import settings  # noqa: E402
from classifier import GptClassifier  # noqa: E402
from db import build_store  # noqa: E402
from engine import run_search  # noqa: E402
from models import TimeWindow  # noqa: E402
from testing.mock_providers import CORPUS, MockClassifier, MockDiscoveryClient  # noqa: E402


def main() -> int:
    log.info("supabase configured=%s | openai configured=%s | serper configured=%s",
             settings.storage_configured, settings.openai_configured, settings.serp_configured)
    assert settings.storage_configured, "SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY missing"
    assert settings.openai_configured, "OPENAI_API_KEY missing"

    store = build_store(settings)
    real_classifier = GptClassifier(settings.openai_api_key, model=settings.openai_model,
                                    timeout_seconds=settings.openai_timeout_seconds,
                                    max_retries=settings.openai_max_retries)
    discovery = MockDiscoveryClient()

    # 1) Direct GPT-4o check against the §2 traps (deterministic verdicts expected).
    log.info("-- real GPT-4o classification of hand-picked traps/genuine posts --")

    def find(needle: str):
        for p in CORPUS:
            if needle.lower() in p.text.lower():
                return p
        raise AssertionError(f"corpus missing a probe for {needle!r}")

    from discovery.base import RawPost
    probes_spec = [
        # (corpus post, expect_reject)
        (find("looking for a freelance video editor to help cut our case studies"), False),  # genuine buyer
        (find("Our company needs a video editing agency or freelancer for a rebrand"), False),  # genuine hiring buyer
        (find("We offer video editing services — book a call"), True),   # seller/offering
        (find("Our agency specializes in video editing for B2B"), True),  # agency self-promotion
        (find("open to work, portfolio in comments"), True),              # job seeker
        (find("5 video editing tips that will double your retention"), True),  # thought leadership
    ]
    probes = [RawPost(post_url=p.url, text=p.text, author_name=p.author,
                      author_profile_url=p.author_url, posted_at=None)
              for p, _ in probes_spec]
    results = real_classifier.classify_batch(
        probes, service="a video editor", lead_type="need_freelancer", country="",
        max_concurrency=min(8, len(probes)))
    mismatches = 0
    for (post, expect_reject), cl in zip(probes_spec, results):
        if cl is None:
            log.warning("classifier dropped %s", post.url)
            mismatches += 1
            continue
        got_reject = cl.lead_type == "irrelevant" or not cl.is_qualified
        flag = "OK " if got_reject == expect_reject else "MISMATCH"
        if got_reject != expect_reject:
            mismatches += 1
        log.info("[%s] %s -> type=%s qualified=%s | %s", flag, post.url[-12:],
                 cl.lead_type, cl.is_qualified, cl.reason)
    if mismatches:
        log.error("%d classifier mismatches — investigate before trusting live prompts", mismatches)
        return 1

    # 2) Full engine run -> real Supabase persistence.
    log.info("-- full engine run (real GPT-4o + real Supabase) --")
    t0 = time.time()
    row = store.create_search(service="a video editor", country="United States",
                              lead_type="need_freelancer", time_window="7d", leads_needed=3)
    summary = run_search(row["id"], store=store, discovery=discovery,
                         classifier=real_classifier, settings=settings)
    log.info("engine summary: status=%s accepted=%d found=%d scanned=%d (%.1fs)",
             summary.status, summary.accepted, summary.found, summary.scanned, time.time() - t0)
    assert summary.status == "completed"
    assert summary.accepted == 3, summary

    leads = store.list_leads(search_id=row["id"])
    assert len(leads) == 3, leads
    for lead in leads:
        log.info("lead: score=%s type=%s date=%s author=%s url=%s",
                 lead["overall_quality_score"], lead["lead_type"], lead["post_date"],
                 lead["author_name"], lead["post_url"])
        assert lead["post_url"].startswith("http")
        assert lead["lead_type"] in {"need_freelancer", "hiring_buyer", "our_agency"}
        assert lead["intent_strength"] in {"explicit", "active_search", "recommendation"}

    patched = store.patch_lead(leads[0]["id"], status="contacted", notes="live e2e")
    assert patched and patched["status"] == "contacted"
    log.info("lead %s patched -> %s", leads[0]["id"], patched["status"])

    used = store.searches_used_today()
    log.info("search rows used today=%d", used)

    # 3) cleanup test rows so the user starts with a clean database.
    from discovery.base import canonical_post_url
    store.patch_lead(leads[0]["id"], status="new", notes=None)
    # delete via management REST is out of scope here; rows are tiny and labeled.
    log.info("LIVE E2E OK — rows persisted in project %s (search %s)",
             settings.supabase_url, row["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
