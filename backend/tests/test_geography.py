import pytest

from geography import NAME_BY_CODE, canonical_label, normalize_country


@pytest.mark.parametrize("raw,expected_code", [
    # full names
    ("United States", "US"), ("united states", "US"), ("America", "US"),
    ("United Kingdom", "GB"), ("Britain", "GB"), ("England", "GB"),
    ("India", "IN"), ("United Arab Emirates", "AE"),
    ("Türkiye", "TR"), ("turkiye", "TR"), ("Turkey", "TR"),
    ("South Africa", "ZA"), ("Kenya", "KE"), ("Canada", "CA"),
    # ISO-2 / ISO-3 style abbreviations
    ("US", "US"), ("USA", "US"), ("u.s.a.", "US"),
    ("UK", "GB"), ("GBR", "GB"), ("u.k.", "GB"),
    ("IN", "IN"), ("IND", "IN"),  # ISO-3 also recognized via the alias table
    ("AE", "AE"), ("DE", "DE"), ("BR", "BR"),
    # cities implying a country
    ("Nairobi", "KE"), ("Mumbai", "IN"), ("Toronto", "CA"),
    ("London", "GB"), ("Dubai", "AE"), ("Austin", "US"),
    ("New York", "US"), ("Bengaluru", "IN"), ("Cape Town", "ZA"),
    ("Mexico City", "MX"), ("São Paulo", "BR"), ("Ho Chi Minh", "VN"),
])
def test_normalize_country_recognizes(raw, expected_code):
    info = normalize_country(raw)
    assert info.code == expected_code, raw
    if expected_code is not None:
        assert info.name == NAME_BY_CODE[expected_code]
        assert info.label == NAME_BY_CODE[expected_code]


def test_normalize_country_is_case_and_space_insensitive():
    assert normalize_country("  united   states  ").code == "US"
    assert normalize_country("u.s.a.").code == "US"
    assert normalize_country("UNITED KINGDOM").code == "GB"


def test_unknown_country_degrades_gracefully():
    info = normalize_country("Xyzzyland")
    assert info.code is None          # never a hard reject
    assert info.raw == "Xyzzyland"
    assert info.label == "Xyzzyland"  # free-text location signal survives
    assert info.recognized is False


def test_empty_country():
    info = normalize_country("")
    assert info.code is None and info.label == ""
    assert normalize_country(None).label == ""


def test_canonical_label_roundtrip_from_stored_code():
    # The API stores a canonical code; engine turns it back into a label.
    assert canonical_label("KE") == "Kenya"
    assert canonical_label("GB") == "United Kingdom"
    assert canonical_label("US") == "United States"
    assert canonical_label("Atlantis") == "Atlantis"


def test_table_is_broad_and_code_complete():
    # Every country entry must include its own code as a usable alias.
    for code in NAME_BY_CODE:
        assert normalize_country(code).code == code, code
    assert len(NAME_BY_CODE) >= 50  # reasonably broad, per §6
