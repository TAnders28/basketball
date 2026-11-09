"""NBA abbreviations. Los Angeles is ambiguous unless the nickname or alias says which club."""

import re

# abbr, city, nickname, extra aliases
_ROWS = [
    ("ATL", "Atlanta", "Hawks", ()),
    ("BOS", "Boston", "Celtics", ()),
    ("BKN", "Brooklyn", "Nets", ("BRK",)),
    ("CHA", "Charlotte", "Hornets", ("CHO",)),
    ("CHI", "Chicago", "Bulls", ()),
    ("CLE", "Cleveland", "Cavaliers", ("Cavs",)),
    ("DAL", "Dallas", "Mavericks", ("Mavs",)),
    ("DEN", "Denver", "Nuggets", ()),
    ("DET", "Detroit", "Pistons", ()),
    ("GSW", "Golden State", "Warriors", ("GS",)),
    ("HOU", "Houston", "Rockets", ()),
    ("IND", "Indiana", "Pacers", ()),
    ("LAC", "Los Angeles", "Clippers", ("LA Clippers", "Los Angeles C", "L.A. Clippers", "Los Angeles Clippers")),
    ("LAL", "Los Angeles", "Lakers", ("LA Lakers", "Los Angeles L", "L.A. Lakers", "Los Angeles Lakers")),
    ("MEM", "Memphis", "Grizzlies", ()),
    ("MIA", "Miami", "Heat", ()),
    ("MIL", "Milwaukee", "Bucks", ()),
    ("MIN", "Minnesota", "Timberwolves", ("Wolves",)),
    ("NOP", "New Orleans", "Pelicans", ("NO", "NOLA")),
    ("NYK", "New York", "Knicks", ("NY",)),
    ("OKC", "Oklahoma City", "Thunder", ()),
    ("ORL", "Orlando", "Magic", ()),
    ("PHI", "Philadelphia", "76ers", ("Sixers", "76ers")),
    ("PHX", "Phoenix", "Suns", ("PHO",)),
    ("POR", "Portland", "Trail Blazers", ("Blazers", "Trail Blazers")),
    ("SAC", "Sacramento", "Kings", ()),
    ("SAS", "San Antonio", "Spurs", ("SA",)),
    ("TOR", "Toronto", "Raptors", ()),
    ("UTA", "Utah", "Jazz", ()),
    ("WAS", "Washington", "Wizards", ("WSH",)),
]


def _Key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


_BY_KEY = {}
_CITY_KEYS = {}
for abbr, city, nickname, aliases in _ROWS:
    _CITY_KEYS.setdefault(_Key(city), set()).add(abbr)
    # Bare city is added later, and only when one club uses it.
    keys = [abbr, nickname, f"{city} {nickname}", *aliases]
    for key in keys:
        _BY_KEY[_Key(key)] = abbr
for city_key, options in _CITY_KEYS.items():
    if len(options) == 1:
        _BY_KEY[city_key] = next(iter(options))


def ResolveTeam(name) -> dict:
    """Return {abbr} or {abbr: None, reason}. Never guesses between the two Los Angeles clubs."""
    if name is None or str(name).strip() == "":
        return {"abbr": None, "reason": "missing_team"}
    key = _Key(str(name))
    if key in _BY_KEY:
        return {"abbr": _BY_KEY[key], "reason": None}
    if key in _CITY_KEYS:
        options = _CITY_KEYS[key]
        if len(options) == 1:
            return {"abbr": next(iter(options)), "reason": None}
        return {"abbr": None, "reason": "ambiguous_team", "input": str(name), "candidates": sorted(options)}
    return {"abbr": None, "reason": "unknown_team", "input": str(name)}
