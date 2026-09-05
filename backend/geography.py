"""Country normalization (§0 / §6).

The user types a free-text "Country / Location" — a full name, an ISO code,
a common abbreviation, or a city that implies a country. This module maps
whatever they typed to a canonical ISO-3166 alpha-2 code and a canonical
display name, case-insensitively, from a broad alias table.

Unknown input degrades gracefully: the caller keeps the raw text as a
free-text location signal instead of hard-rejecting (a typo must never
silently zero out every result).

Nothing here is service-specific; it is pure geography.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# code -> (display name, aliases). Keep this broad; it is a lookup table,
# not an if/else chain, and not an allow-list — unknown input is kept as-is.
_COUNTRIES: dict[str, tuple[str, tuple[str, ...]]] = {
    # --- English-speaking markets -----------------------------------------
    "US": ("United States", ("united states", "usa", "us", "america", "united states of america", "the usa", "u.s.a.", "u.s.")),
    "GB": ("United Kingdom", ("united kingdom", "uk", "gb", "great britain", "britain", "england", "scotland", "wales", "northern ireland", "u.k.", "gbr")),
    "CA": ("Canada", ("canada", "ca", "can")),
    "AU": ("Australia", ("australia", "au", "aus", "aussie")),
    "NZ": ("New Zealand", ("new zealand", "nz", "nzl")),
    "IE": ("Ireland", ("ireland", "ie", "irl", "eire")),
    # --- Asia / Middle East -------------------------------------------------
    "IN": ("India", ("india", "in", "ind", "bharat")),
    "AE": ("United Arab Emirates", ("united arab emirates", "uae", "ae", "are", "dubai", "abu dhabi", "sharjah")),
    "SG": ("Singapore", ("singapore", "sg", "sgp")),
    "HK": ("Hong Kong", ("hong kong", "hk", "hkg")),
    "PK": ("Pakistan", ("pakistan", "pk", "pak")),
    "BD": ("Bangladesh", ("bangladesh", "bd", "bgd")),
    "LK": ("Sri Lanka", ("sri lanka", "lk", "lka", "ceylon")),
    "NP": ("Nepal", ("nepal", "np", "npl")),
    "PH": ("Philippines", ("philippines", "ph", "phl", "phil")),
    "MY": ("Malaysia", ("malaysia", "my", "mys")),
    "ID": ("Indonesia", ("indonesia", "id", "idn")),
    "TH": ("Thailand", ("thailand", "th", "tha")),
    "VN": ("Vietnam", ("vietnam", "vn", "vnm")),
    "CN": ("China", ("china", "cn", "chn", "prc")),
    "TW": ("Taiwan", ("taiwan", "tw", "twn", "roc")),
    "JP": ("Japan", ("japan", "jp", "jpn")),
    "KR": ("South Korea", ("south korea", "korea", "kr", "kor", "republic of korea")),
    "IL": ("Israel", ("israel", "il", "isr")),
    "QA": ("Qatar", ("qatar", "qa", "qat")),
    "SA": ("Saudi Arabia", ("saudi arabia", "sa", "sau", "ksa")),
    "KW": ("Kuwait", ("kuwait", "kw", "kwt")),
    "OM": ("Oman", ("oman", "om", "omn")),
    "BH": ("Bahrain", ("bahrain", "bh", "bhr")),
    "TR": ("Türkiye", ("turkey", "türkiye", "turkiye", "tr", "tur")),
    # --- Europe --------------------------------------------------------------
    "DE": ("Germany", ("germany", "de", "deu", "deutschland")),
    "FR": ("France", ("france", "fr", "fra")),
    "ES": ("Spain", ("spain", "es", "esp", "españa")),
    "IT": ("Italy", ("italy", "it", "ita", "italia")),
    "NL": ("Netherlands", ("netherlands", "nl", "nld", "holland")),
    "BE": ("Belgium", ("belgium", "be", "bel")),
    "CH": ("Switzerland", ("switzerland", "ch", "che", "swiss")),
    "AT": ("Austria", ("austria", "at", "aut")),
    "SE": ("Sweden", ("sweden", "se", "swe")),
    "NO": ("Norway", ("norway", "no", "nor")),
    "DK": ("Denmark", ("denmark", "dk", "dnk")),
    "FI": ("Finland", ("finland", "fi", "fin")),
    "PL": ("Poland", ("poland", "pl", "pol")),
    "PT": ("Portugal", ("portugal", "pt", "prt")),
    "CZ": ("Czech Republic", ("czech republic", "czechia", "cz", "cze")),
    "GR": ("Greece", ("greece", "gr", "grc")),
    "RO": ("Romania", ("romania", "ro", "rou")),
    "UA": ("Ukraine", ("ukraine", "ua", "ukr")),
    "RU": ("Russia", ("russia", "ru", "rus", "russian federation")),
    # --- Middle East (Iran) + Africa -------------------------------------------
    "ZA": ("South Africa", ("south africa", "za", "zaf", "rsa")),
    "NG": ("Nigeria", ("nigeria", "ng", "nga")),
    "KE": ("Kenya", ("kenya", "ke", "ken")),
    "GH": ("Ghana", ("ghana", "gh", "gha")),
    "EG": ("Egypt", ("egypt", "eg", "egy")),
    "MA": ("Morocco", ("morocco", "ma", "mar")),
    "TZ": ("Tanzania", ("tanzania", "tz", "tza")),
    "UG": ("Uganda", ("uganda", "ug", "uga")),
    "ET": ("Ethiopia", ("ethiopia", "et", "eth")),
    # --- Americas --------------------------------------------------------------
    "BR": ("Brazil", ("brazil", "br", "bra", "brasil")),
    "MX": ("Mexico", ("mexico", "mx", "mex")),
    "AR": ("Argentina", ("argentina", "ar", "arg")),
    "CL": ("Chile", ("chile", "cl", "chl")),
    "CO": ("Colombia", ("colombia", "co", "col")),
    "PE": ("Peru", ("peru", "pe", "per")),
}

# Major cities that strongly imply a country (city name -> country code).
_CITY_TO_CODE: dict[str, str] = {
    "new york": "US", "los angeles": "US", "san francisco": "US", "austin": "US",
    "chicago": "US", "seattle": "US", "miami": "US", "boston": "US", "denver": "US",
    "london": "GB", "manchester": "GB", "edinburgh": "GB", "birmingham": "GB",
    "toronto": "CA", "vancouver": "CA", "montreal": "CA", "ottawa": "CA",
    "sydney": "AU", "melbourne": "AU", "brisbane": "AU", "perth": "AU",
    "auckland": "NZ", "wellington": "NZ",
    "dublin": "IE", "cork": "IE",
    "mumbai": "IN", "delhi": "IN", "new delhi": "IN", "bangalore": "IN", "bengaluru": "IN",
    "hyderabad": "IN", "chennai": "IN", "pune": "IN", "gurgaon": "IN",
    "nairobi": "KE", "mombasa": "KE",
    "lagos": "NG", "abuja": "NG",
    "accra": "GH", "johannesburg": "ZA", "cape town": "ZA", "cairo": "EG",
    "dubai": "AE", "abu dhabi": "AE", "doha": "QA", "riyadh": "SA", "kuwait city": "KW",
    "singapore": "SG", "hong kong": "HK", "kuala lumpur": "MY", "jakarta": "ID",
    "bangkok": "TH", "manila": "PH", "ho chi minh": "VN", "beijing": "CN",
    "shanghai": "CN", "taipei": "TW", "tokyo": "JP", "osaka": "JP", "seoul": "KR",
    "tel aviv": "IL", "istanbul": "TR", "berlin": "DE", "munich": "DE",
    "paris": "FR", "madrid": "ES", "barcelona": "ES", "rome": "IT", "milan": "IT",
    "amsterdam": "NL", "brussels": "BE", "zurich": "CH", "geneva": "CH",
    "vienna": "AT", "stockholm": "SE", "oslo": "NO", "copenhagen": "DK",
    "helsinki": "FI", "warsaw": "PL", "lisbon": "PT", "prague": "CZ",
    "athens": "GR", "bucharest": "RO", "kyiv": "UA", "kiev": "UA", "moscow": "RU",
    "sao paulo": "BR", "rio de janeiro": "BR", "mexico city": "MX", "buenos aires": "AR",
    "santiago": "CL", "bogota": "CO", "lima": "PE",
}

_WS = re.compile(r"\s+")


def _fold_key(raw: str) -> str:
    """Case/space-insensitive key: lowercase, accent-folded (Türkiye ->
    turkiye, São -> sao), ASCII punctuation dropped (U.S.A. -> usa)."""
    text = unicodedata.normalize("NFKD", raw or "")
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
    return _WS.sub(" ", text).strip()


_ALIAS_TO_CODE: dict[str, str] = {}
for _code, (_name, _aliases) in _COUNTRIES.items():
    for _alias in _aliases:
        _ALIAS_TO_CODE[_fold_key(_alias)] = _code
    _ALIAS_TO_CODE[_fold_key(_name)] = _code
for _city, _code in _CITY_TO_CODE.items():
    _ALIAS_TO_CODE[_fold_key(_city)] = _code

NAME_BY_CODE: dict[str, str] = {code: name for code, (name, _) in _COUNTRIES.items()}


@dataclass(frozen=True, slots=True)
class CountryInfo:
    raw: str                 # exactly what the user typed
    code: str | None         # canonical ISO alpha-2 when recognized, else None
    name: str | None         # canonical display name when recognized, else None

    @property
    def recognized(self) -> bool:
        return self.code is not None

    @property
    def label(self) -> str:
        """Human-friendly value for the classifier/scoring: canonical name
        when recognized, otherwise the raw free text (graceful fallback)."""
        return self.name or self.raw or ""


def normalize_country(raw: str | None) -> CountryInfo:
    """Map free-text country/location input to a canonical code + name."""
    text = (raw or "").strip()
    if not text:
        return CountryInfo(raw="", code=None, name=None)
    code = _ALIAS_TO_CODE.get(_fold_key(text))
    return CountryInfo(
        raw=text,
        code=code,
        name=NAME_BY_CODE.get(code) if code else None,
    )


def canonical_label(raw: str | None) -> str:
    return normalize_country(raw).label
