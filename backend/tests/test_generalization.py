"""§0 generalization: the SAME pipeline (queries -> discovery -> prefilter ->
classifier -> gates) must work for unrelated services and countries purely
from the service/country values passed in — zero code changes."""
from datetime import UTC, datetime

import pytest

from config import Settings
from db import MemoryStore
from engine import run_search
from models import LeadType, TimeWindow
from testing.mock_providers import MockClassifier, MockDiscoveryClient, build_service_corpus

# (service typed by the user, corpus terms, country as typed, canonical label embedded in posts)
CASES = [
    ("a plumber", ("plumber",), "Nairobi", "Kenya"),
    ("a UX designer", ("UX", "designer"), "Toronto", "Canada"),
    ("a wedding photographer", ("wedding", "photographer"), "Mumbai", "India"),
]


@pytest.mark.parametrize("service,terms,typed_country,country_label", CASES)
def test_pipeline_generalizes_across_unrelated_services(service, terms, typed_country, country_label):
    corpus = build_service_corpus(terms, location=country_label)
    requested = LeadType.NEED_FREELANCER
    expected = sum(1 for p in corpus if p.lead_type == requested)  # strict type
    assert expected == 3  # corpus sanity: 3 need_freelancer buyers per service

    store = MemoryStore()
    sid = store.create_search(
        service=service, country=typed_country, lead_type=requested.value,
        time_window="28d", leads_needed=100,
    )["id"]
    summary = run_search(
        sid, store=store,
        discovery=MockDiscoveryClient(corpus=corpus),
        classifier=MockClassifier(corpus=corpus),
        settings=Settings(),
    )
    assert summary.status == "completed"
    assert summary.accepted == expected, (service, summary)

    leads = store.list_leads(search_id=sid)
    assert len(leads) == expected
    cutoff_date = TimeWindow("28d").cutoff(datetime.now(UTC)).date()
    for lead in leads:
        # Every accepted lead MUST carry a real post URL (identity + dedupe).
        assert lead["post_url"].startswith("https://www.linkedin.com/posts/gen-")
        assert lead["author_name"]
        assert lead["author_profile_url"].startswith("https://")
        assert lead["post_date"] >= cutoff_date
        assert lead["overall_quality_score"] >= 60  # cleared the gate
        assert lead["lead_type"] == requested.value  # strict: only the requested type


@pytest.mark.parametrize("service,terms,typed_country,country_label", CASES)
def test_pipeline_rejects_traps_for_any_service(service, terms, typed_country, country_label):
    """Traps (seller, job seeker, marketplace, job ad, thought leadership)
    must never surface as leads for any service — and neither may genuine
    buyers of a DIFFERENT lead type."""
    corpus = build_service_corpus(terms, location=country_label)
    requested = LeadType.HIRING_BUYER
    expected = sum(1 for p in corpus if p.lead_type == requested)  # 2 hiring buyers
    store = MemoryStore()
    sid = store.create_search(
        service=service, country=typed_country, lead_type=requested.value,
        time_window="28d", leads_needed=100,
    )["id"]
    run_search(sid, store=store, discovery=MockDiscoveryClient(corpus=corpus),
               classifier=MockClassifier(corpus=corpus), settings=Settings())
    leads = store.list_leads(search_id=sid)
    assert len(leads) == expected  # never a trap AND never a wrong-type buyer
    for lead in leads:
        assert lead["lead_type"] == requested.value
