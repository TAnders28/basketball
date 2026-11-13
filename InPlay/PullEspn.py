r"""Pull ESPN scoreboards and compact per-play win probability + clock. Resumable.

ESPN homeWinPercentage is ESPN's model, not a tradable price.

    .venv\Scripts\python.exe -m InPlay.PullEspn --seasons 2027 2026 2025 2024 2023 2022 2021 2020 2019 2018
Season N = the season ending in year N (2027 = 2026-27).
"""

import argparse
import os
import sys
from datetime import date, timedelta

from InPlay.Http import GetJson, HttpError, ReadJson, WriteJsonAtomic

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ROOT = os.path.join("data", "inplay", "espn")
KEEP_TYPES = {2, 3, 5}  # regular, postseason, play-in


def SeasonDates(season, today=None):
    today = today or date.today()
    day, end = date(season - 1, 10, 1), min(date(season, 6, 30), today - timedelta(days=1))
    while day <= end:
        yield day
        day += timedelta(days=1)


def ScoreboardGames(body):
    out = []
    for event in body.get("events", []):
        season_type = (event.get("season") or {}).get("type")
        comp = (event.get("competitions") or [{}])[0]
        status = ((comp.get("status") or {}).get("type") or {})
        if season_type not in KEEP_TYPES or not status.get("completed"):
            continue
        out.append({"id": event["id"], "date": event.get("date"), "season_type": season_type})
    return out


def _Abbr(comp, side):
    for c in comp.get("competitors", []):
        if c.get("homeAway") == side:
            return c
    return {}


def CompactSummary(body):
    """Keep plays (seq, period, clock, score, wallclock, type, team) and the WP array."""
    header = body.get("header") or {}
    comp = (header.get("competitions") or [{}])[0]
    home, away = _Abbr(comp, "home"), _Abbr(comp, "away")
    plays = []
    for p in body.get("plays", []):
        plays.append({
            "id": str(p.get("id")), "seq": int(p.get("sequenceNumber") or 0),
            "period": (p.get("period") or {}).get("number"),
            "clock": (p.get("clock") or {}).get("displayValue"),
            "hs": p.get("homeScore"), "as": p.get("awayScore"),
            "wall": p.get("wallclock"), "type": (p.get("type") or {}).get("text"),
            "team": (p.get("team") or {}).get("id"), "scoring": bool(p.get("scoringPlay")),
            "sv": p.get("scoreValue"), "shoot": bool(p.get("shootingPlay")),
        })
    wp = [{"play": str(w.get("playId")), "h": w.get("homeWinPercentage")} for w in body.get("winprobability", [])]
    pick = []
    for pc in body.get("pickcenter", []) or []:
        pick.append({"provider": (pc.get("provider") or {}).get("name"), "spread": pc.get("spread"),
                     "overUnder": pc.get("overUnder"),
                     "homeML": ((pc.get("homeTeamOdds") or {}).get("moneyLine")),
                     "awayML": ((pc.get("awayTeamOdds") or {}).get("moneyLine"))})
    return {
        "id": header.get("id"), "date": comp.get("date"),
        "season": (header.get("season") or {}).get("year"), "season_type": (header.get("season") or {}).get("type"),
        "home": (home.get("team") or {}).get("abbreviation"), "away": (away.get("team") or {}).get("abbreviation"),
        "home_id": (home.get("team") or {}).get("id"), "away_id": (away.get("team") or {}).get("id"),
        "home_score": home.get("score"), "away_score": away.get("score"),
        "home_won": home.get("winner"), "plays": plays, "wp": wp, "pickcenter": pick,
    }


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    args = ap.parse_args(argv)
    for season in args.seasons:
        games = []
        for day in SeasonDates(season):
            key = day.strftime("%Y%m%d")
            path = os.path.join(ROOT, "scoreboard", f"{key}.json")
            if os.path.exists(path):
                games += ReadJson(path)
                continue
            try:
                found = ScoreboardGames(GetJson(SITE + "/scoreboard", params={"dates": key, "limit": 50}))
            except HttpError as exc:
                print(f"ERR scoreboard {key} {exc}", flush=True)
                continue
            WriteJsonAtomic(path, found)
            games += found
        print(f"season {season}: {len(games)} games", flush=True)
        done = errors = 0
        for i, g in enumerate(games):
            path = os.path.join(ROOT, "games", str(season), f"{g['id']}.json")
            if os.path.exists(path):
                continue
            try:
                WriteJsonAtomic(path, CompactSummary(GetJson(SITE + "/summary", params={"event": g["id"]})))
                done += 1
            except HttpError as exc:
                errors += 1
                print(f"ERR summary {g['id']} {exc}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"season {season}: {i + 1}/{len(games)} pulled={done} errors={errors}", flush=True)
        print(f"DONE season {season} pulled={done} errors={errors}", flush=True)


if __name__ == "__main__":
    sys.exit(Main())