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

    clf = GptClassifier("sk-fake-for-construction-only", model="deepseek-chat")  # no call is made
    assert clf.provider == "deepseek"
    assert clf.json_mode == "json_object"
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


REJECT_JSON = ('{"lead_type": "irrelevant", "intent_strength": "none", '
               '"is_buying_sourcing": false, "is_selling_offering": true, "is_job_seek": false, '
               '"service_match_score": 10, "commercial_intent_score": 10, "decision_maker_signal": false, '
               '"evidence_strength": 20, "overall_quality_score": 15, "is_qualified": false, '
               '"confidence": 0.9, "evidence": "we offer services", "reason": "seller"}')


class _Choice:
    def __init__(self, content: str) -> None:
        class _Msg:
            pass
        m = _Msg()
        m.content = content
        self.message = m


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _FakeCompletions:
    def __init__(self, captured: dict, content: str) -> None:
        self.captured = captured
        self.content = content

    def create(self, **kwargs):
        self.captured.update(kwargs)
        return _Resp(self.content)


def _stub_client(clf: GptClassifier, captured: dict, content: str = REJECT_JSON) -> None:
    class _Chat:
        completions = None
    class _Client:
        chat = None
    _Client.chat = _Chat()
    _Chat.completions = _FakeCompletions(captured, content)
    clf._client = _Client()


def test_deepseek_request_uses_json_object_mode():
    """DeepSeek (json_object) must NOT send OpenAI's json_schema response_format;
    it must send {"type": "json_object"} against the DeepSeek base URL."""
    from discovery.base import RawPost

    captured: dict = {}
    clf = GptClassifier("sk-deepseek-test", model="deepseek-chat",
                        base_url="https://api.deepseek.com", provider="deepseek")
    _stub_client(clf, captured)
    post = RawPost(post_url="https://www.linkedin.com/posts/1", text="We offer services.",
                   author_name="A", posted_at=None)
    result = clf._classify_one(post, service="video editor", requested_type="need_freelancer", country="")
    assert result is not None and result.lead_type == "irrelevant"
    assert captured["model"] == "deepseek-chat"
    assert captured["response_format"] == {"type": "json_object"}
    texts = " ".join(m["content"] for m in captured["messages"])
    assert "irrelevant" in texts or "lead_type" in texts  # schema spelled out for json_object


def test_openai_request_uses_json_schema_mode():
    from discovery.base import RawPost

    captured: dict = {}
    clf = GptClassifier("sk-openai-test", model="gpt-4o", provider="openai", json_mode="json_schema")
    assert clf.json_mode == "json_schema"
    _stub_client(clf, captured)
    post = RawPost(post_url="https://www.linkedin.com/posts/2", text="irrelevant", author_name=None, posted_at=None)
    result = clf._classify_one(post, service="video editor", requested_type="hiring_buyer", country="")
    assert result is not None
    assert captured["response_format"]["type"] == "json_schema"
    assert captured["model"] == "gpt-4o"
