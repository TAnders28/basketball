r"""
Derive possessions and 10-player stints from PlayByPlayV3 + GameRotation.

Time is measured in tenths of a second from tip (GameRotation's unit): regulation periods are 7200 tenths,
overtimes 3000. A possession is a run of events with the same offensive team; it changes on a defensive
rebound, a turnover, or a made basket / final free throw followed by the other team's action. This is the
usual play-by-play approximation (team rebounds and some dead-ball edge cases are not perfect).

Each possession records: offense/defense team, start/end time, how it started (after a make, a defensive
rebound, a turnover, or period start), points, FGA / 3PA / FTA / offensive rebounds / turnover, whether a
shot came within 8 s of a live-ball start (transition), the assister of a made shot, the on-court fives,
and whether it was "clutch" (last 5 min of the 4th or OT, margin <= 5 at the start).
"""

import re
from collections import defaultdict

REG_PERIOD, OT_PERIOD = 7200, 3000
CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
AST_RE = re.compile(r"\(([^()]+?) \d+ AST\)")
TRANSITION_TENTHS = 80


def Elapsed(period: int, clock: str) -> int:
    m = CLOCK_RE.match(clock or "")
    remaining = int(round((int(m.group(1)) * 60 + float(m.group(2))) * 10)) if m else 0
    if period <= 4:
        return (period - 1) * REG_PERIOD + (REG_PERIOD - remaining)
    return 4 * REG_PERIOD + (period - 5) * OT_PERIOD + (OT_PERIOD - remaining)


def GameLength(max_period: int) -> int:
    return 4 * REG_PERIOD + max(0, max_period - 4) * OT_PERIOD


def Lineups(rotation: dict, home: str, away: str):
    """[(team, player, in, out)] from GameRotation's two frames."""
    out = []
    for side, team in (("home", home), ("away", away)):
        for r in (rotation or {}).get(side, []):
            if r.get("IN_TIME_REAL") is None or r.get("OUT_TIME_REAL") is None:
                continue
            out.append((team, int(r["PERSON_ID"]), float(r["IN_TIME_REAL"]), float(r["OUT_TIME_REAL"])))
    return out


def OnCourt(stints, team: str, t: float) -> tuple:
    return tuple(sorted(p for tm, p, a, b in stints if tm == team and a <= t < b))


def Events(pbp: list, home: str, away: str) -> list:
    """Clean event list: time, team, player, type, made, value, points scored (by event team), score after."""
    events, hs, as_ = [], 0, 0
    for r in pbp or []:
        t = Elapsed(int(r.get("period") or 1), r.get("clock"))
        typ = r.get("actionType") or ""
        team = r.get("teamTricode") or ""
        if not team and r.get("location") in ("h", "v"):
            team = home if r["location"] == "h" else away   # team-level rows (timeouts, team rebounds/turnovers)
        pts = 0
        sh, sa = r.get("scoreHome"), r.get("scoreAway")
        if sh not in (None, "") and sa not in (None, ""):
            nh, na = int(sh), int(sa)
            pts = (nh - hs) if team == home else (na - as_) if team == away else 0
            hs, as_ = nh, na
        desc = r.get("description") or ""
        made = (r.get("shotResult") == "Made") or (typ == "Free Throw" and not desc.upper().startswith("MISS"))
        assist = None
        if typ == "Made Shot":
            m = AST_RE.search(desc)
            assist = m.group(1).strip() if m else None
        events.append({"t": t, "period": int(r.get("period") or 1), "team": team, "player": int(r.get("personId") or 0),
                       "name": r.get("playerName") or "", "type": typ, "sub": r.get("subType") or "", "made": made,
                       "value": int(r.get("shotValue") or 0), "dist": r.get("shotDistance") or 0, "pts": pts,
                       "home_score": hs, "away_score": as_, "assist_name": assist, "desc": desc})
    return events


OFFENSE_TYPES = {"Made Shot", "Missed Shot", "Free Throw", "Turnover"}


def BuildPossessions(pbp: list, rotation: dict, home: str, away: str) -> list:
    events = Events(pbp, home, away)
    stints = Lineups(rotation, home, away)
    names = defaultdict(dict)   # team -> last name -> player id (to resolve assisters)
    for e in events:
        if e["team"] and e["player"]:
            names[e["team"]][e["name"]] = e["player"]
    other = {home: away, away: home}
    poss, cur, last_shot_team, start_type, start_t = [], None, None, "period_start", None
    current_period = 1

    def Open(team, t, how, e):
        t = start_t if start_t is not None and start_t <= t and how != "period_start" else t
        margin = e["home_score"] - e["away_score"]
        clutch = (t >= 4 * REG_PERIOD - 3000 or e["period"] > 4) and abs(margin) <= 5
        return {"offense": team, "defense": other[team], "start": t, "end": t, "start_type": how, "pts": 0,
                "fga": 0, "fg3a": 0, "fta": 0, "oreb": 0, "tov": 0, "first_shot_t": None, "assisted": False,
                "assister": None, "scorer": None, "clutch": clutch, "margin_start_home": margin,
                "off5": OnCourt(stints, team, t), "def5": OnCourt(stints, other[team], t), "shots": []}

    for e in events:
        if e["period"] != current_period:
            if cur:
                poss.append(cur)
            cur, start_type, current_period, start_t = None, "period_start", e["period"], None
        typ, team = e["type"], e["team"]
        if typ in OFFENSE_TYPES and team in other:
            if cur is None or cur["offense"] != team:
                if cur:
                    poss.append(cur)
                cur = Open(team, e["t"], start_type, e)
            cur["end"] = e["t"]
            if typ in ("Made Shot", "Missed Shot"):
                cur["fga"] += 1
                cur["fg3a"] += e["value"] == 3
                last_shot_team = team
                if cur["first_shot_t"] is None:
                    cur["first_shot_t"] = e["t"]
                cur["shots"].append({"player": e["player"], "value": e["value"], "made": e["made"], "dist": e["dist"], "sub": e["sub"]})
                if typ == "Made Shot":
                    cur["scorer"] = e["player"]
                    if e["assist_name"]:
                        cur["assisted"] = True
                        cur["assister"] = names[team].get(e["assist_name"].split(" ")[-1]) or names[team].get(e["assist_name"])
                    start_type, start_t = "after_make", e["t"]
            elif typ == "Free Throw":
                cur["fta"] += 1
                last_shot_team = team if not e["made"] else last_shot_team
                start_type, start_t = "after_make", e["t"]
            elif typ == "Turnover":
                cur["tov"] += 1
                start_type = "after_steal" if "steal" in e["desc"].lower() or "STL" in e["desc"] else "after_turnover"
                start_t = e["t"]
            cur["pts"] += max(e["pts"], 0)
        elif typ == "Rebound" and team in other:
            if cur and team == cur["offense"]:
                # player rebounds only: team "rebounds" after a missed first free throw are bookkeeping rows
                cur["oreb"] += 1 if 0 < e["player"] < 1610612700 else 0
            else:
                start_type, start_t = "after_dreb", e["t"]
                if cur:
                    poss.append(cur)
                cur = Open(team, e["t"], "after_dreb", e)
        elif team in other and e["pts"] > 0 and cur and team == cur["offense"]:
            cur["pts"] += e["pts"]
    if cur:
        poss.append(cur)
    for p in poss:
        live = p["start_type"] in ("after_dreb", "after_steal", "after_turnover")
        p["transition"] = bool(live and p["first_shot_t"] is not None and p["first_shot_t"] - p["start"] <= TRANSITION_TENTHS)
    return poss


def BuildStints(pbp: list, rotation: dict, home: str, away: str, possessions: list = None) -> list:
    """Intervals with constant on-court fives: seconds, points and possessions per side."""
    stints = Lineups(rotation, home, away)
    if not stints:
        return []
    events = Events(pbp, home, away)
    end_t = GameLength(max([e["period"] for e in events] or [4]))
    cuts = sorted({0.0, float(end_t)} | {a for _, _, a, _ in stints} | {b for _, _, _, b in stints})
    possessions = possessions if possessions is not None else BuildPossessions(pbp, rotation, home, away)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2
        row = {"start": a, "end": b, "seconds": (b - a) / 10.0, "home5": OnCourt(stints, home, mid),
               "away5": OnCourt(stints, away, mid), "home_pts": 0, "away_pts": 0, "home_poss": 0, "away_poss": 0}
        for e in events:
            if a < e["t"] <= b and e["pts"] > 0:
                row["home_pts" if e["team"] == home else "away_pts"] += e["pts"] if e["team"] in (home, away) else 0
        for p in possessions:
            if a <= p["start"] < b:
                row["home_poss" if p["offense"] == home else "away_poss"] += 1
        out.append(row)
    return out


def GameEvents(pbp: list, home: str, away: str) -> dict:
    """Coaching-style counts per team: timeouts, coach's challenges, technicals, fouls."""
    out = {home: defaultdict(int), away: defaultdict(int)}
    for e in Events(pbp, home, away):
        team = e["team"]
        if team not in out:
            continue
        if e["type"] == "Timeout" and "challenge" in (e["sub"] + e["desc"]).lower():
            out[team]["challenges"] += 1
        elif e["type"] == "Timeout":
            out[team]["timeouts"] += 1
        elif e["type"] == "Instant Replay" and "challenge" in (e["sub"] + e["desc"]).lower():
            out[team]["challenges"] += 1
        elif e["type"] == "Foul":
            out[team]["fouls"] += 1
            if "technical" in e["sub"].lower():
                out[team]["technicals"] += 1
    return {k: dict(v) for k, v in out.items()}
