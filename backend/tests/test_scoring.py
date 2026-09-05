import pytest

from models import IntentStrength, LeadClassification
from scoring import ScoreConfig, compute_score


def _cl(lead_type="need_freelancer", intent="explicit", match=90.0, commercial=85.0,
         decision=True, evidence=80.0, buying=True, selling=False, job_seek=False,
         qualified=True, conf=0.95):
    return LeadClassification(
        lead_type=lead_type,
        intent_strength=intent,
        is_buying_sourcing=buying,
        is_selling_offering=selling,
        is_job_seek=job_seek,
        service_match_score=match,
        commercial_intent_score=commercial,
        decision_maker_signal=decision,
        evidence_strength=evidence,
        overall_quality_score=90.0,
        is_qualified=qualified,
        confidence=conf,
        evidence="looking for a video editor",
        reason="buyer",
    )


def test_strong_need_freelancer_passes_all_gates():
    cfg = ScoreConfig()
    r = compute_score(_cl(), requested_country="", post_text="looking for a video editor", cfg=cfg)
    assert r.keep
    assert all(r.gates.values())
    # overall = .3*100(intent) + .25*90(match) + .15*85(commercial) + .1*100(decision)
    #           + .1*100(location: no country constraint) + .1*80(evidence) = 93.25
    assert r.overall == pytest.approx(93.25, abs=0.3)


def test_recommendation_level_is_the_floor():
    # intent = recommendation (70) still clears the intent gate.
    assert compute_score(_cl(intent="recommendation")).keep
    # problem_awareness (50) is below the floor and must be rejected.
    r = compute_score(_cl(intent="problem_awareness"))
    assert not r.keep
    assert r.gates["intent_ok"] is False


def test_irrelevant_type_rejected():
    r = compute_score(_cl(lead_type="irrelevant", buying=False))
    assert not r.keep
    assert r.gates["type_is_buyer"] is False


def test_seller_in_disguise_rejected_even_when_model_waffles():
    # Agency self-promotion is a seller, never our_agency.
    r = compute_score(_cl(lead_type="our_agency", buying=False, selling=True))
    assert not r.keep
    assert r.gates["buying_not_selling"] is False


def test_job_seeker_rejected():
    r = compute_score(_cl(intent="none", match=70.0, job_seek=True, buying=False))
    assert not r.keep
    assert r.gates["buying_not_selling"] is False


def test_service_match_gate_blocks_adjacent_craft():
    r = compute_score(_cl(match=40.0))
    assert not r.keep
    assert r.gates["service_match_ok"] is False


def test_model_qualified_verdict_is_a_gate():
    r = compute_score(_cl(qualified=False))
    assert not r.keep
    assert r.gates["model_qualified"] is False


def test_low_confidence_does_not_auto_reject():
    # Confidence is reported, not gated — evidence is.
    assert compute_score(_cl(conf=0.6)).keep


def test_location_confidence_country_hit_vs_unknown():
    r_hit = compute_score(_cl(), requested_country="Texas", post_text="We shoot in Austin, Texas.")
    r_unknown = compute_score(_cl(), requested_country="Germany", post_text="Looking for a video editor.")
    assert r_hit.keep and r_unknown.keep
    assert r_hit.overall > r_unknown.overall


def test_minimum_intent_strength_configurable():
    cfg = ScoreConfig(min_intent_strength="explicit")
    r = compute_score(_cl(intent="active_search"), cfg=cfg)
    assert not r.keep
    assert r.gates["intent_ok"] is False
    assert compute_score(_cl(intent="explicit"), cfg=cfg).keep
