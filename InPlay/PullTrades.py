r"""Kalshi public trade prints (second timestamps, taker side) for a sample of home-team game markets.

Read-only, public endpoints, rate-limited by InPlay.Http (0.25 s/request), resumable (skips files already written).
Sample: every Kalshi tune game (2025 playoffs), every forward game (2026-27), and a seeded random 200 holdout games.
Trades only from 5 h before close to close. Uses /historical/trades before the 2026-09-13 cutoff, else /markets/trades.

    .venv\Scripts\python.exe -m InPlay.PullTrades
"""

import os
import random
import sys
from datetime import datetime, timezone

from InPlay.Http import GetJson, HttpError, ReadJson, WriteJsonAtomic

B = "https://api.elections.kalshi.com/trade-api/v2"
OUT = os.path.join("data", "inplay", "kalshi", "trades")
CUTOFF = "2026-09-13"


def Ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def Sample(ms, n_hold=200, seed=12):
    home = [m for m in ms if m.get("is_home")]
    tune = [m for m in home if m["date_et"] < "2025-07-01"]
    fwd = [m for m in home if m["date_et"] >= "2026-10-01"]
    hold = [m for m in home if "2025-10-01" <= m["date_et"] < "2026-07-01"]
    random.Random(seed).shuffle(hold)
    return tune + fwd + sorted(hold[:n_hold], key=lambda m: m["date_et"])


def Compact(t):
    return [round(Ts(t["created_time"]), 3), float(t["yes_price_dollars"]), float(t["count_fp"]), 1 if t["taker_side"] == "yes" else -1]


def PullOne(m):
    path = os.path.join(OUT, m["ticker"] + ".json")
    if os.path.exists(path):
        return 0
    end = Ts(m["close_time"])
    start = end - 5 * 3600
    url = B + ("/historical/trades" if m["date_et"] < CUTOFF else "/markets/trades")
    rows, cursor, pages = [], None, 0
    while True:
        p = {"ticker": m["ticker"], "limit": 1000, "max_ts": int(end) + 60}
        if cursor:
            p["cursor"] = cursor
        r = GetJson(url, params=p)
        pages += 1
        got = r.get("trades") or []
        rows += [Compact(t) for t in got]
        cursor = r.get("cursor")
        if not got or not cursor or Ts(got[-1]["created_time"]) < start or pages > 400:
            break
    rows = sorted(x for x in rows if start <= x[0] <= end + 60)
    WriteJsonAtomic(path, {"ticker": m["ticker"], "date_et": m["date_et"], "trades": rows})
    return len(rows)


def Main():
    ms = ReadJson(os.path.join("data", "inplay", "kalshi", "markets.json"))
    todo = Sample(ms)
    print(f"{len(todo)} markets", flush=True)
    err = 0
    for k, m in enumerate(todo):
        try:
            n = PullOne(m)
        except HttpError as e:
            err += 1
            print("ERR", m["ticker"], e, flush=True)
            continue
        if k % 20 == 0:
            print(f"{k}/{len(todo)} {m['ticker']} trades={n} errors={err}", flush=True)
    print(f"DONE errors={err}", flush=True)


if __name__ == "__main__":
    sys.exit(Main())