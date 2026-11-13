"""Pure alignment helpers: game clock, monotone wallclock, as-of joins, team codes, seasons.

Every join here is as-of (last value at or before t), so nothing from after t leaks in.
"""

import bisect
import calendar
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from LiveOdds.Teams import ResolveTeam

EASTERN = ZoneInfo("America/New_York")
EXTRA = {"UTAH": "UTA", "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "WSH": "WAS", "PHO": "PHX",
         "BRK": "BKN", "CHO": "CHA", "NJ": "BKN", "NOH": "NOP"}


def Team(code):
    if code is None:
        return None
    code = str(code).upper()
    if code in EXTRA:
        return EXTRA[code]
    return ResolveTeam(code)["abbr"]


def IsoTs(text):
    """ISO UTC ('2026-10-20T19:07:39Z', '2026-10-20T19:00Z', '+00') -> unix seconds."""
    if not text:
        return None
    s = str(text).replace("Z", "").replace("+00:00", "").replace("+00", "").replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
        try:
            return calendar.timegm(datetime.strptime(s, fmt).timetuple())
        except ValueError:
            continue
    return None


def EtDate(ts):
    return datetime.fromtimestamp(ts, timezone.utc).astimezone(EASTERN).strftime("%Y-%m-%d")


def SeasonOf(ts):
    """Season label = calendar year the season ends (2026 = 2025-26). Aug/Sep belong to the next season."""
    d = datetime.fromtimestamp(ts, timezone.utc).astimezone(EASTERN)
    return d.year + 1 if d.month >= 8 else d.year


def ClockSeconds(text):
    """'11:48' -> 708; '0.0' / '24.3' -> seconds as float."""
    if text is None:
        return None
    s = str(text).strip()
    try:
        if ":" in s:
            mm, ss = s.split(":", 1)
            return int(mm) * 60 + float(ss)
        return float(s)
    except ValueError:
        return None


def Elapsed(period, clock_text):
    """Game seconds elapsed. Regulation 4x720, each OT 300."""
    rem = ClockSeconds(clock_text)
    if period is None or rem is None:
        return None
    period = int(period)
    if period <= 4:
        return (period - 1) * 720 + (720 - rem)
    return 2880 + (period - 5) * 300 + (300 - rem)


def MonotoneWall(plays):
    """Plays sorted by sequence; wallclock made non-decreasing (cumulative max). Missing walls inherit."""
    out, run = [], None
    for p in sorted(plays, key=lambda x: x["seq"]):
        ts = IsoTs(p.get("wall"))
        if ts is not None:
            run = ts if run is None else max(run, ts)
        out.append((run, p))
    return [(t, p) for t, p in out if t is not None]


def AsOf(times, values, t):
    """Last value with time <= t, or None. times must be sorted ascending."""
    i = bisect.bisect_right(times, t) - 1
    return values[i] if i >= 0 else None


def EspnStates(game):
    """Per-play state rows (wall_ts, elapsed, period, home_score, away_score, wp_home) in sequence order."""
    wp = {w["play"]: w["h"] for w in game.get("wp", []) if w.get("h") is not None}
    rows = []
    for ts, p in MonotoneWall(game.get("plays", [])):
        el = Elapsed(p.get("period"), p.get("clock"))
        if el is None:
            continue
        rows.append({"t": ts, "el": el, "per": int(p["period"]), "hs": p.get("hs") or 0, "as": p.get("as") or 0,
                     "wp": wp.get(p["id"]), "type": p.get("type"), "team": p.get("team")})
    # forward-fill wp in sequence
    last = None
    for r in rows:
        if r["wp"] is None:
            r["wp"] = last
        last = r["wp"]
    return rows


def TipAndEnd(states):
    if not states:
        return None, None
    return states[0]["t"], states[-1]["t"]


def KalshiMid(bid, ask):
    """Mid of the YES book if both sides are real quotes, else None."""
    if bid is None or ask is None or bid <= 0 or ask >= 1 or ask < bid:
        return None
    return 0.5 * (bid + ask)