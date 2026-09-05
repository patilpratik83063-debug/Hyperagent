import time

import pytest
from fastapi.testclient import TestClient

import main as main_mod
from testing.mock_providers import MockClassifier, MockDiscoveryClient


class MissingDiscovery:
    name = "unconfigured"
    config_errors = ["SERPER_API_KEY is not set"]

    def search_posts(self, queries, since, *, results_per_query=25):
        raise RuntimeError("should never be called")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main_mod, "discovery", MockDiscoveryClient())
    monkeypatch.setattr(main_mod, "classifier", MockClassifier())
    monkeypatch.setattr(main_mod, "store", main_mod.build_store(main_mod.settings))
    return TestClient(main_mod.app)


def _wait_terminal(client, search_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/search/{search_id}/status")
        assert r.status_code == 200
        s = r.json()
        if s["status"] in ("completed", "failed", "no_results"):
            return s
        time.sleep(0.3)
    raise AssertionError("search did not finish in time")


def test_health_and_usage_endpoints(client):
    assert client.get("/health").json() == {"status": "ok"}
    usage = client.get("/api/usage").json()
    assert set(usage) == {"used", "limit", "remaining", "resets_in_seconds"}
    assert usage["remaining"] >= 0


def test_search_runs_end_to_end_and_returns_exactly_n(client):
    resp = client.post("/api/search", json={
        "service": "a video editor",
        "lead_type": "need_freelancer",
        "time_window": "7d",
        "leads_needed": 4,   # small enough that the offline corpus guarantees it
        "country": "",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "queued"
    search_id = body["search_id"]

    final = _wait_terminal(client, search_id)
    assert final["status"] == "completed"
    assert final["accepted"] == 4  # exactly N

    leads = client.get(f"/api/leads?search_id={search_id}").json()
    assert leads["count"] == 4
    assert all(l["lead_type"] == "need_freelancer" for l in leads["leads"])

    # Re-filter saved leads by a fresh window (posted_at-based, no re-run).
    filtered = client.get(f"/api/leads?search_id={search_id}&time_window=24h").json()
    assert filtered["count"] <= leads["count"]
    # Every lead has the fields the UI renders.
    first = leads["leads"][0]
    assert first["post_url"].startswith("http")
    assert first["author_name"]
    assert first["overall_quality_score"] is not None

    # PATCH a lead: status + notes persist.
    lead_id = first["id"]
    patched = client.patch(f"/api/leads/{lead_id}", json={"status": "contacted", "notes": "called back"}).json()
    assert patched["status"] == "contacted"
    assert patched["notes"] == "called back"
    reloaded = client.get(f"/api/leads?search_id={search_id}").json()["leads"]
    row = next(l for l in reloaded if l["id"] == lead_id)
    assert row["status"] == "contacted" and row["notes"] == "called back"

    # Invalid window value is rejected cleanly.
    bad = client.get(f"/api/leads?search_id={search_id}&time_window=99d")
    assert bad.status_code == 422


def test_shortage_never_pads_beyond_qualified(client):
    # A 24h window can only surface the corpus' genuinely qualified posts from
    # that window; the engine must stop there and never pad to 100.
    from datetime import UTC, datetime

    from models import LeadType, TimeWindow
    from testing.mock_providers import CORPUS, _now, _service_words

    cutoff = TimeWindow("24h").cutoff(datetime.now(UTC))
    words = _service_words("video editor")
    expected = sum(
        1 for p in CORPUS
        if p.lead_type == LeadType.NEED_FREELANCER  # strict: requested type only
        and _now(p.days_ago) >= cutoff
        and any(w in p.text.lower() for w in words)
    )
    assert expected > 0  # corpus sanity: the window does contain qualified posts

    resp = client.post("/api/search", json={
        "service": "a video editor",
        "lead_type": "need_freelancer",
        "time_window": "24h",
        "leads_needed": 100,
    })
    assert resp.status_code == 200, resp.text
    search_id = resp.json()["search_id"]
    final = _wait_terminal(client, search_id)
    assert final["status"] == "completed"
    assert final["accepted"] == expected  # exactly the qualified ones — no padding
    assert client.get(f"/api/leads?search_id={search_id}").json()["count"] == expected


def test_unconfigured_discovery_blocks_search(monkeypatch):
    monkeypatch.setattr(main_mod, "discovery", MissingDiscovery())
    monkeypatch.setattr(main_mod, "classifier", MockClassifier())
    resp = TestClient(main_mod.app).post("/api/search", json={
        "service": "video editor", "lead_type": "need_freelancer", "leads_needed": 10,
    })
    assert resp.status_code == 503
    assert "SERPER_API_KEY" in resp.text


def test_validation_errors(client):
    resp = client.post("/api/search", json={"service": "x", "lead_type": "need_freelancer"})
    assert resp.status_code == 422
    resp = client.post("/api/search", json={"service": "video editor", "lead_type": "not_a_type", "leads_needed": 10})
    assert resp.status_code == 422
    assert client.get("/api/search/does-not-exist/status").status_code == 404
    assert client.patch("/api/leads/does-not-exist", json={"status": "new"}).status_code == 404


def test_past_searches_lists_rows(client):
    client.post("/api/search", json={
        "service": "video editor", "lead_type": "need_freelancer", "time_window": "7d", "leads_needed": 10,
    })
    resp = client.get("/api/searches").json()
    assert resp["count"] >= 1
    assert resp["searches"][0]["lead_type"] == "need_freelancer"


def test_country_is_normalized_to_canonical_code_on_store(client):
    resp = client.post("/api/search", json={
        "service": "video editor",
        "country": "United States",  # full name, not an ISO code
        "lead_type": "need_freelancer",
        "time_window": "7d",
        "leads_needed": 10,
    })
    assert resp.status_code == 200, resp.text
    status = client.get(f"/api/search/{resp.json()['search_id']}/status").json()
    assert status["country"] == "US"

    # City implying a country + graceful fallback for unknown free text.
    r2 = client.post("/api/search", json={
        "service": "video editor", "country": "Nairobi",
        "lead_type": "need_freelancer", "leads_needed": 10,
    }).json()
    assert client.get(f"/api/search/{r2['search_id']}/status").json()["country"] == "KE"
    r3 = client.post("/api/search", json={
        "service": "video editor", "country": "Xyzzyland",
        "lead_type": "need_freelancer", "leads_needed": 10,
    }).json()
    assert client.get(f"/api/search/{r3['search_id']}/status").json()["country"] == "Xyzzyland"


def test_frontend_served_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Hyperclients" in resp.text
    assert "/api/search" in resp.text
    assert "Copy URL" in resp.text  # post URL is a first-class, copyable field
