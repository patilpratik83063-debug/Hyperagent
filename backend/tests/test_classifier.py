import pytest

from classifier import ClassifierConfigError, GptClassifier, parse_and_validate
from models import LeadClassification

GOOD = {
    "lead_type": "need_freelancer",
    "intent_strength": "explicit",
    "is_buying_sourcing": True,
    "is_selling_offering": False,
    "is_job_seek": False,
    "service_match_score": 90.0,
    "commercial_intent_score": 80.0,
    "decision_maker_signal": True,
    "evidence_strength": 85.0,
    "overall_quality_score": 88.0,
    "is_qualified": True,
    "confidence": 0.95,
    "evidence": "looking for a video editor",
    "reason": "clear buyer",
}


def test_parse_and_validate_accepts_valid_payload():
    cl = parse_and_validate(GOOD)
    assert isinstance(cl, LeadClassification)
    assert cl.lead_type == "need_freelancer"


def test_parse_and_validate_accepts_json_string():
    import json

    cl = parse_and_validate(json.dumps(GOOD))
    assert cl is not None and cl.is_qualified is True


@pytest.mark.parametrize("mutator", [
    lambda d: d | {"lead_type": "garbage"},
    lambda d: {k: v for k, v in d.items() if k != "reason"},
    lambda d: d | {"service_match_score": 150},
    lambda d: d | {"intent_strength": "very_explicit"},
])
def test_parse_and_validate_rejects_bad_payloads(mutator):
    assert parse_and_validate(mutator(dict(GOOD))) is None
    assert parse_and_validate("not json {") is None
    assert parse_and_validate([]) is None
    assert parse_and_validate(None) is None


def test_classifier_requires_api_key():
    clf = GptClassifier("")
    assert clf.config_errors  # fail-closed: surfaced before any search runs
    with pytest.raises(ClassifierConfigError):
        clf.classify_batch([], service="x", lead_type="need_freelancer")


def test_batch_isolates_per_post_failures(monkeypatch):
    from discovery.base import RawPost

    clf = GptClassifier("sk-fake-for-construction-only")  # no call is made
    posts = [RawPost(post_url="https://x/1", text="ok", posted_at=None),
             RawPost(post_url="https://x/2", text="boom", posted_at=None)]

    def fake_one(self, post, *, service, requested_type, country):
        if post.post_url.endswith("/2"):
            raise RuntimeError("transient")
        return parse_and_validate(GOOD)

    monkeypatch.setattr(GptClassifier, "_classify_one", fake_one)
    results = clf.classify_batch(posts, service="video editor", lead_type="need_freelancer", max_concurrency=2)
    assert results[0] is not None   # survived
    assert results[1] is None       # dropped, never guessed, batch did not die
