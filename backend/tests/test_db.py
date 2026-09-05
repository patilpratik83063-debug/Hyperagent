"""Supabase serialization + store protocol regression tests (no network)."""
from datetime import UTC, date, datetime

from db import MemoryStore, SupabaseStore


def test_jsonable_serializes_dates_and_datetimes():
    assert SupabaseStore._jsonable(date(2026, 9, 5)) == "2026-09-05"
    assert SupabaseStore._jsonable(datetime(2026, 9, 5, 10, 30, tzinfo=UTC)) == "2026-09-05T10:30:00+00:00"
    # Naive datetime must still serialize (no astimezone crash).
    assert SupabaseStore._jsonable(datetime(2026, 9, 5, 10, 30)).endswith("+00:00")
    nested = {"post_date": date(2026, 9, 5), "meta": {"at": datetime(2026, 9, 5, 12, 0, tzinfo=UTC)},
              "tags": [date(2026, 1, 1)], "n": 3, "flag": None}
    out = SupabaseStore._jsonable(nested)
    assert out["post_date"] == "2026-09-05"
    assert out["meta"]["at"] == "2026-09-05T12:00:00+00:00"
    assert out["tags"] == ["2026-01-01"]
    assert out["n"] == 3 and out["flag"] is None


def test_memory_store_full_protocol_roundtrip():
    store = MemoryStore()
    row = store.create_search(service="a plumber", country="KE", lead_type="need_freelancer",
                              time_window="7d", leads_needed=2)
    sid = row["id"]
    assert store.get_search(sid)["status"] == "queued"

    inserted = store.insert_leads_many([
        {"search_id": sid, "lead_type": "need_freelancer", "time_window": "7d",
         "post_url": "https://www.linkedin.com/posts/1", "author_name": "A",
         "post_date": date(2026, 9, 1), "overall_quality_score": 91.0,
         "service_match_score": 90.0, "intent_strength": "explicit"},
        {"search_id": sid, "lead_type": "hiring_buyer", "time_window": "7d",
         "post_url": "https://www.linkedin.com/posts/2", "author_name": "B",
         "post_date": date(2026, 9, 2), "overall_quality_score": 88.0,
         "service_match_score": 85.0, "intent_strength": "active_search"},
    ])
    assert inserted == 2
    # Duplicate post_url is silently skipped (identity constraint).
    assert store.insert_leads_many([
        {"search_id": sid, "lead_type": "need_freelancer", "post_url": "https://www.linkedin.com/posts/1",
         "post_date": date(2026, 9, 1)},
    ]) == 0
    assert len(store.list_leads(search_id=sid)) == 2
    store.update_search(sid, status="completed", accepted_count=2, finished_at=datetime.now(UTC))
    assert store.get_search(sid)["status"] == "completed"
    assert store.patch_lead(store.list_leads(search_id=sid)[0]["id"], status="contacted")["status"] == "contacted"
    assert store.searches_used_today() >= 1
    assert len(store.list_searches()) >= 1
