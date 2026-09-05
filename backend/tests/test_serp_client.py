"""Serper/Google-SERP discovery client tests — HTTP fully mocked, no network."""
from datetime import UTC, datetime

import pytest

import discovery.serp_client as serp_mod
from discovery.base import DiscoveryConfigError
from discovery.serp_client import SerperDiscoveryClient


class FakeResp:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or ""

    def json(self):
        return self._payload


def _client(**kw):
    return SerperDiscoveryClient("test-serper-key", results_per_query=5, **kw)


def test_requires_api_key():
    with pytest.raises(DiscoveryConfigError):
        SerperDiscoveryClient("")
    assert _client().config_errors == []


def test_full_query_appends_site_and_fresh_after_date():
    client = _client()
    since = datetime(2025, 6, 10, 12, 0, 0, tzinfo=UTC)
    q = client._full_query('looking for a plumber -"we offer"', since)
    assert q.startswith('looking for a plumber -"we offer"')
    assert "site:linkedin.com/posts" in q
    assert "after:2025-06-10" in q  # computed from `since` at call time


def test_maps_organic_results_and_parses_author(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["payload"] = json
        captured["url"] = url
        return FakeResp(payload={
            "organic": [
                {
                    "link": "https://www.linkedin.com/posts/jane-123-abc",
                    "title": "Jane Smith on LinkedIn: We are looking for a video editor for our launch",
                    "snippet": "We are looking for a video editor for our product launch. Budget is ready.",
                    "date": "Aug 24, 2026",
                },
                {
                    "link": "https://www.linkedin.com/posts/john-456-def",
                    "title": "Anyone know a good plumber in Nairobi?",
                    "snippet": "Anyone know a good plumber in Nairobi? Need one this week.",
                    # no date -> kept, engine/UI tolerate missing post dates
                },
            ]
        })

    monkeypatch.setattr(serp_mod.httpx, "post", fake_post)
    client = _client()
    since = datetime(2025, 6, 1, tzinfo=UTC)
    batch = client.search_posts(['looking for a video editor', 'need a plumber'], since)
    assert captured["payload"]["num"] == 5
    assert "site:linkedin.com/posts" in captured["payload"]["q"]
    assert "after:2025-06-01" in captured["payload"]["q"]
    assert captured["url"] == "https://google.serper.dev/search"

    posts = {p.post_url: p for p in batch.posts}
    assert len(posts) == 2
    jane = posts["https://www.linkedin.com/posts/jane-123-abc"]
    assert jane.author_name == "Jane Smith"
    assert jane.posted_at is not None and jane.posted_at.year == 2026 and jane.posted_at.month == 8
    assert "video editor" in jane.text
    john = posts["https://www.linkedin.com/posts/john-456-def"]
    assert john.posted_at is None  # undated results are kept, not silently dropped
    assert batch.provider_errors == []


def test_dedupes_and_filters_non_post_results(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResp(payload={"organic": [
            {"link": "https://www.linkedin.com/posts/same-post", "title": "t", "snippet": "a"},
            {"link": "https://www.linkedin.com/posts/same-post?trk=abc", "title": "t2", "snippet": "a2"},
            {"link": "https://example.com/blog/video-editors", "title": "not a post", "snippet": "x"},
        ]})

    monkeypatch.setattr(serp_mod.httpx, "post", fake_post)
    batch = _client().search_posts(["q1", "q2"], datetime(2025, 1, 1, tzinfo=UTC))
    assert len(batch.posts) == 1
    assert batch.posts[0].post_url == "https://www.linkedin.com/posts/same-post"


def test_zero_results_is_quiet_normal_behavior(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResp(payload={"organic": []})

    monkeypatch.setattr(serp_mod.httpx, "post", fake_post)
    batch = _client().search_posts(["tight phrase query"], datetime(2025, 1, 1, tzinfo=UTC))
    assert batch.posts == []           # expected SERP behavior, not an error
    assert batch.provider_errors == []
    assert batch.queries_used == ["tight phrase query"]


def test_provider_errors_are_loud_and_isolated(monkeypatch):
    calls = {"n": 0}

    def fake_post(url, headers, json, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResp(status_code=401, payload={}, text="unauthorized")
        if calls["n"] == 2:
            return FakeResp(payload={"error": "Rate limit exceeded"})
        return FakeResp(payload={"organic": []})

    monkeypatch.setattr(serp_mod.httpx, "post", fake_post)
    since = datetime(2025, 1, 1, tzinfo=UTC)
    batch = _client().search_posts(["q1", "q2", "q3"], since)
    assert len(batch.provider_errors) == 2  # failed queries isolated, batch survives
    assert "401" in batch.provider_errors[0]
    assert "Rate limit" in batch.provider_errors[1]
    assert batch.posts == []


def test_http_exception_recorded_per_query(monkeypatch):
    def boom(url, headers, json, timeout):
        raise serp_mod.httpx.HTTPError("connection reset")

    monkeypatch.setattr(serp_mod.httpx, "post", boom)
    batch = _client().search_posts(["q"], datetime(2025, 1, 1, tzinfo=UTC))
    assert batch.provider_errors and "connection reset" in batch.provider_errors[0]
