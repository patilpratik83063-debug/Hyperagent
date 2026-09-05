from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from models import LeadType, SearchRequest, TimeWindow


def test_lead_type_values():
    assert LeadType.NEED_FREELANCER.value == "need_freelancer"
    assert LeadType.HIRING_BUYER.value == "hiring_buyer"
    assert LeadType.OUR_AGENCY.value == "our_agency"
    assert len(LeadType) == 3


@pytest.mark.parametrize(
    "window,days",
    [("24h", 1), ("7d", 7), ("14d", 14), ("28d", 28)],
)
def test_time_window_days(window, days):
    assert TimeWindow(window).days() == days


def test_cutoff_is_computed_fresh_and_rounded_down():
    now = datetime(2025, 6, 15, 14, 30, 0, tzinfo=UTC)
    # 7 days before 15 Jun 14:30 is 8 Jun 14:30 -> rounded down to 8 Jun 00:00 UTC
    assert TimeWindow.DAYS_7.cutoff(now) == datetime(2025, 6, 8, 0, 0, 0, tzinfo=UTC)
    # Guarantees at least one full day of coverage and >= the window itself.
    assert TimeWindow.DAYS_7.cutoff(now) <= now - timedelta(days=7)
    # 24h window also lands on a UTC midnight boundary.
    assert TimeWindow.HOURS_24.cutoff(now) == datetime(2025, 6, 14, 0, 0, 0, tzinfo=UTC)


def test_after_iso_is_never_hardcoded():
    a = TimeWindow.DAYS_14.after_iso(datetime(2025, 1, 10, 8, 0, tzinfo=UTC))
    b = TimeWindow.DAYS_14.after_iso(datetime(2025, 6, 10, 8, 0, tzinfo=UTC))
    assert a == "2024-12-27"
    assert b == "2025-05-27"
    assert a != b  # computed from "now" at call time, never stored once


def test_search_request_validation():
    req = SearchRequest(service="a video editor", lead_type=LeadType.HIRING_BUYER,
                        time_window=TimeWindow.DAYS_7, leads_needed=25)
    assert req.service == "a video editor"
    assert req.leads_needed == 25

    with pytest.raises(ValidationError):
        SearchRequest(service="ab", lead_type=LeadType.HIRING_BUYER)  # too short
    with pytest.raises(ValidationError):
        SearchRequest(service="ok service", lead_type="bogus_type")
    with pytest.raises(ValidationError):
        SearchRequest(service="ok service", lead_type=LeadType.HIRING_BUYER, leads_needed=0)
    with pytest.raises(ValidationError):
        SearchRequest(service="ok service", lead_type=LeadType.HIRING_BUYER, leads_needed=501)
