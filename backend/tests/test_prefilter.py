from prefilter import prefilter

BUYER = (
    "We just wrapped a launch and are looking for a freelance video editor to help cut case studies. "
    "DM if you know someone great."
)
AGENCY_BUYER = "We're a production agency looking for freelancers to work with our agency on client projects."
FREELANCE_HIRE = "We are hiring a freelance video editor for a 6-week campaign, not a full-time role."


def test_genuine_buyers_are_kept():
    assert prefilter(BUYER).keep
    assert prefilter(AGENCY_BUYER).keep


def test_buyer_with_job_ad_words_is_kept_for_classifier():
    # Ambiguous (buyer marker + 'full-time') -> classifier owns the call.
    v = prefilter(FREELANCE_HIRE)
    assert v.keep
    assert v.matched["buyer"] is True


def test_job_ad_dropped_without_buyer_signal():
    v = prefilter("We are hiring a full-time video editor. Apply now at careers.example.com — vacancy #4421.")
    assert not v.keep
    assert v.dropped_for == "job_ad"


def test_seller_dropped_without_buyer_signal():
    v = prefilter("We offer video editing services — book a call for a free consultation and a quote within 24h.")
    assert not v.keep
    assert v.dropped_for == "seller"


def test_job_seeker_dropped():
    v = prefilter("I'm a freelance video editor available for projects — open to work, portfolio in comments.")
    assert not v.keep
    assert v.dropped_for == "job_seeker"


def test_marketplace_dropped():
    v = prefilter("Our platform connects brands with vetted video editors. Join our talent network today — free.")
    assert not v.keep
    assert v.dropped_for == "marketplace"


def test_agency_self_promo_dropped():
    v = prefilter("Our agency specializes in video editing for B2B. We can help your brand stand out.")
    assert not v.keep
    assert v.dropped_for == "seller"


def test_thought_leadership_dropped():
    v = prefilter("5 video editing tips that will double your retention. Save this for later 🚀")
    assert not v.keep
    assert v.dropped_for == "thought_leadership"


def test_neutral_advice_without_strong_markers_kept_for_ai():
    # Not cheap-droppable: no buyer marker but no near-certain negative either.
    v = prefilter("In my experience, good video editing is 80% storytelling.")
    assert v.keep
