r"""Pull settled Kalshi KXNBASPREAD markets and 1-minute candles for forward-split games (live API only). Resumable.

Only the most-traded strikes per game (top N by volume) are pulled, to stay polite.

    .venv\Scripts\python.exe -m InPlay.PullDerivs --per-game 8
"""

import argparse
import os
import sys
from collections import defaultdict

from InPlay.Http import GetJson, HttpError, ReadJson, WriteJsonAtomic
from InPlay.PullKalshi import BASE, IsoToTs, NormalizeCandle

ROOT = os.path.join("data", "inplay", "kalshi", "derivs")


def Enumerate(series, min_date="2026-09-13", max_date="2026-11-11", get=GetJson):
    rows, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "settled", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        body = get(BASE + "/markets", params=params)
        for m in body.get("markets", []):
            ev = m.get("event_ticker", "")
            rows.append(dict(ticker=m["ticker"], event=ev, sub=m.get("yes_sub_title"), strike=m.get("floor_strike"),
                             vol=float(m.get("volume_fp") or 0), result=m.get("result"), close_time=m.get("close_time")))
        cursor = body.get("cursor")
        print(f"{series}: {len(rows)}", flush=True)
        if not cursor or not body.get("markets"):
            break
    return rows


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--series", default="KXNBASPREAD")
    a = ap.parse_args(argv)
    lp = os.path.join(ROOT, a.series + "_markets.json")
    if not os.path.exists(lp):
        WriteJsonAtomic(lp, Enumerate(a.series))
    rows = ReadJson(lp)
    games = {r["event_ticker"].replace("KXNBAGAME", a.series) for r in ReadJson(os.path.join("data", "inplay", "kalshi", "markets.json"))
             if "2026-09-13" <= r["date_et"] <= "2026-11-11"}
    by = defaultdict(list)
    for r in rows:
        if r["event"] in games:
            by[r["event"]].append(r)
    pick = [m for ev in sorted(by) for m in sorted(by[ev], key=lambda m: -m["vol"])[:a.per_game] if m["vol"] > 0]
    print(f"{len(by)} events, {len(pick)} markets to pull", flush=True)
    done = err = 0
    for k, m in enumerate(pick):
        p = os.path.join(ROOT, "candles", m["ticker"] + ".json")
        if os.path.exists(p):
            continue
        end = IsoToTs(m["close_time"])
        try:
            body = GetJson(f"{BASE}/series/{a.series}/markets/{m['ticker']}/candlesticks",
                           params={"start_ts": end - 5 * 3600, "end_ts": end, "period_interval": 1})
            WriteJsonAtomic(p, {"ticker": m["ticker"], "meta": m, "candles": [NormalizeCandle(c) for c in body.get("candlesticks", [])]})
            done += 1
        except HttpError as exc:
            err += 1
            print("ERR", m["ticker"], exc, flush=True)
        if (k + 1) % 100 == 0:
            print(f"{k+1}/{len(pick)} done={done} err={err}", flush=True)
    print(f"DONE done={done} err={err}", flush=True)


if __name__ == "__main__":
    sys.exit(Main())
