r"""Pull closed Polymarket NBA single-game moneylines and ~1-minute CLOB price history. Resumable.

    .venv\Scripts\python.exe -m InPlay.PullPolymarket
"""

import argparse
import calendar
import json
import os
import re
import sys
from datetime import datetime

from InPlay.Http import GetJson, HttpError, ReadJson, WriteJsonAtomic
from LiveOdds.Teams import ResolveTeam

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
ROOT = os.path.join("data", "inplay", "polymarket")
SLUG_RE = re.compile(r"^nba-([a-z]{2,4})-([a-z]{2,4})-(\d{4}-\d{2}-\d{2})$")


def ParseStart(text):
    if not text:
        return None
    text = text.replace("T", " ").replace("Z", "+00")
    return calendar.timegm(datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").timetuple())


def MarketRow(raw):
    """Keep a game moneyline whose two outcomes resolve to the slug's teams. Home = second slug code."""
    slug = raw.get("slug") or ""
    m = SLUG_RE.match(slug)
    if not m or raw.get("sportsMarketType") != "moneyline":
        return None
    try:
        outcomes = json.loads(raw.get("outcomes") or "[]")
        tokens = json.loads(raw.get("clobTokenIds") or "[]")
        prices = json.loads(raw.get("outcomePrices") or "[]")
    except ValueError:
        return None
    if len(outcomes) != 2 or len(tokens) != 2:
        return None
    abbrs = [ResolveTeam(o)["abbr"] for o in outcomes]
    away = ResolveTeam(m.group(1).upper())["abbr"]
    home = ResolveTeam(m.group(2).upper())["abbr"]
    if None in abbrs or home not in abbrs or away not in abbrs or home == away:
        return None
    hi = abbrs.index(home)
    start = ParseStart(raw.get("gameStartTime"))
    if start is None:
        return None
    home_won = None
    if len(prices) == 2:
        try:
            ph = float(prices[hi])
            home_won = 1 if ph > 0.99 else 0 if ph < 0.01 else None
        except ValueError:
            pass
    return {"slug": slug, "date_slug": m.group(3), "away": away, "home": home, "home_token": tokens[hi],
            "start_ts": start, "home_won": home_won, "volume": raw.get("volume"), "id": raw.get("id")}


def Enumerate(get=GetJson):
    rows, cursor, pages = {}, None, 0
    while True:
        params = {"tag_id": 745, "closed": "true", "sports_market_types": "moneyline", "limit": 500}
        if cursor:
            params["after_cursor"] = cursor
        body = get(GAMMA + "/markets/keyset", params=params)
        for raw in body.get("markets", []):
            row = MarketRow(raw)
            if row:
                rows[row["slug"]] = row
        pages += 1
        cursor = body.get("next_cursor")
        print(f"page {pages}: {len(rows)} game moneylines", flush=True)
        if not cursor or not body.get("markets") or pages > 200:
            break
    return sorted(rows.values(), key=lambda r: r["start_ts"])


def PullHistory(row, get=GetJson):
    body = get(CLOB + "/prices-history", params={"market": row["home_token"], "startTs": row["start_ts"] - 3600,
                                                 "endTs": row["start_ts"] + 4 * 3600, "fidelity": 1})
    return [{"t": int(p["t"]), "p": float(p["p"])} for p in body.get("history", [])]


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-list", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    list_path = os.path.join(ROOT, "markets.json")
    if args.refresh_list or not os.path.exists(list_path):
        WriteJsonAtomic(list_path, Enumerate())
    rows = ReadJson(list_path)
    print(f"{len(rows)} game moneylines", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    todo = [r for r in rows if not os.path.exists(os.path.join(ROOT, "prices", r["slug"] + ".json"))]
    if args.limit:
        todo = todo[: args.limit]

    def One(row):
        try:
            WriteJsonAtomic(os.path.join(ROOT, "prices", row["slug"] + ".json"),
                            {"slug": row["slug"], "history": PullHistory(row)})
            return True
        except HttpError as exc:
            print(f"ERR {row['slug']} {exc}", flush=True)
            return False

    done = errors = 0
    # 4 in flight; Http still enforces the per-host minimum gap (<= 4 requests/s).
    with ThreadPoolExecutor(max_workers=4) as ex:
        for k, ok in enumerate(ex.map(One, todo)):
            done += ok
            errors += not ok
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(todo)} pulled={done} errors={errors}", flush=True)
    print(f"DONE pulled={done} errors={errors}", flush=True)


if __name__ == "__main__":
    sys.exit(Main())