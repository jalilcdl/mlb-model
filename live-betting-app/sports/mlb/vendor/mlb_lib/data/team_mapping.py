"""
Canonical team codes and cross-source abbreviation mapping.

Different data sources use different abbreviation conventions:
  - MLB Stats API (statsapi.mlb.com) uses one set of abbreviations + numeric team IDs.
  - Baseball-Reference (scraped via pybaseball's schedule_and_record) uses a slightly
    different set (e.g. "CHW" vs "CWS", "KCR" vs "KC", "SDP" vs "SD", "SFG" vs "SF",
    "TBR" vs "TB", "WSN" vs "WSH") and it changes historically for relocated franchises
    (the Athletics played as "OAK" through 2024 and "ATH" from 2025 onward).

We use the MLB Stats API abbreviation as the canonical team code everywhere in this
project, and translate to/from Baseball-Reference codes only at the data-loading edge.
"""

# code -> (statsapi team id, statsapi abbr == code, full name, bref abbr (current/default))
_TEAMS = [
    (109, "ARI", "Arizona Diamondbacks", "ARI"),
    (144, "ATL", "Atlanta Braves", "ATL"),
    (110, "BAL", "Baltimore Orioles", "BAL"),
    (111, "BOS", "Boston Red Sox", "BOS"),
    (112, "CHC", "Chicago Cubs", "CHC"),
    (145, "CWS", "Chicago White Sox", "CHW"),
    (113, "CIN", "Cincinnati Reds", "CIN"),
    (114, "CLE", "Cleveland Guardians", "CLE"),
    (115, "COL", "Colorado Rockies", "COL"),
    (116, "DET", "Detroit Tigers", "DET"),
    (117, "HOU", "Houston Astros", "HOU"),
    (118, "KC", "Kansas City Royals", "KCR"),
    (108, "LAA", "Los Angeles Angels", "LAA"),
    (119, "LAD", "Los Angeles Dodgers", "LAD"),
    (146, "MIA", "Miami Marlins", "MIA"),
    (158, "MIL", "Milwaukee Brewers", "MIL"),
    (142, "MIN", "Minnesota Twins", "MIN"),
    (121, "NYM", "New York Mets", "NYM"),
    (147, "NYY", "New York Yankees", "NYY"),
    (133, "ATH", "Athletics", "ATH"),
    (143, "PHI", "Philadelphia Phillies", "PHI"),
    (134, "PIT", "Pittsburgh Pirates", "PIT"),
    (135, "SD", "San Diego Padres", "SDP"),
    (136, "SEA", "Seattle Mariners", "SEA"),
    (137, "SF", "San Francisco Giants", "SFG"),
    (138, "STL", "St. Louis Cardinals", "STL"),
    (139, "TB", "Tampa Bay Rays", "TBR"),
    (140, "TEX", "Texas Rangers", "TEX"),
    (141, "TOR", "Toronto Blue Jays", "TOR"),
    (120, "WSH", "Washington Nationals", "WSN"),
]

# Per-season overrides where the Baseball-Reference abbreviation differs from the
# current default (franchise relocations / rebrands).
_BREF_SEASON_OVERRIDES = {
    "ATH": {season: "OAK" for season in range(1968, 2025)},  # Oakland Athletics through 2024
}

TEAM_BY_CODE = {abbr: {"id": tid, "code": abbr, "name": name, "bref": bref} for tid, abbr, name, bref in _TEAMS}
TEAM_BY_ID = {tid: rec for tid, rec in ((t["id"], t) for t in TEAM_BY_CODE.values())}
_BREF_TO_CODE_DEFAULT = {t["bref"]: t["code"] for t in TEAM_BY_CODE.values()}


def all_codes():
    return list(TEAM_BY_CODE.keys())


def team_name(code):
    return TEAM_BY_CODE[code]["name"]


def code_from_statsapi_id(team_id):
    return TEAM_BY_ID[int(team_id)]["code"]


def bref_abbr(code, season):
    """Baseball-Reference abbreviation for a canonical code in a given season."""
    overrides = _BREF_SEASON_OVERRIDES.get(code, {})
    if season in overrides:
        return overrides[season]
    return TEAM_BY_CODE[code]["bref"]


def code_from_bref(bref_code, season):
    """Reverse lookup: Baseball-Reference abbreviation + season -> canonical code."""
    for code, seasons in _BREF_SEASON_OVERRIDES.items():
        if seasons.get(season) == bref_code:
            return code
    if bref_code in _BREF_TO_CODE_DEFAULT:
        return _BREF_TO_CODE_DEFAULT[bref_code]
    raise KeyError(f"Unknown Baseball-Reference abbreviation '{bref_code}' for season {season}")


# Club-name / city keywords -> code, for resolving free-text team names from odds
# feeds (e.g. Highlightly returns "Yankees" / "New York Yankees", not "NYY").
_NAME_KEYWORDS = {
    "diamondbacks": "ARI", "d-backs": "ARI", "dbacks": "ARI",
    "braves": "ATL", "orioles": "BAL", "red sox": "BOS", "redsox": "BOS",
    "cubs": "CHC", "white sox": "CWS", "whitesox": "CWS", "reds": "CIN",
    "guardians": "CLE", "indians": "CLE",  # Cleveland: Indians -> Guardians (renamed 2022)
    "rockies": "COL", "tigers": "DET", "astros": "HOU",
    "royals": "KC", "angels": "LAA", "dodgers": "LAD", "marlins": "MIA",
    "brewers": "MIL", "twins": "MIN", "mets": "NYM", "yankees": "NYY",
    "athletics": "ATH", "phillies": "PHI", "pirates": "PIT", "padres": "SD",
    "mariners": "SEA", "giants": "SF", "cardinals": "STL", "rays": "TB",
    "rangers": "TEX", "blue jays": "TOR", "bluejays": "TOR", "nationals": "WSH",
}


def code_from_name(*names):
    """Resolve a canonical code from one or more free-text team names/labels
    (any of Highlightly's awayTeam.name / .displayName, etc.). Tries exact full
    name, then a club-name/city keyword contained in the text. Returns None if
    nothing matches -- callers should surface unresolved names rather than guess
    (see src/data/highlightly.py)."""
    full = {v["name"].lower(): k for k, v in TEAM_BY_CODE.items()}
    for raw in names:
        if not raw:
            continue
        t = str(raw).strip().lower()
        if t in full:
            return full[t]
    for raw in names:
        if not raw:
            continue
        t = str(raw).strip().lower()
        for kw, code in _NAME_KEYWORDS.items():
            if kw in t:
                return code
    return None
