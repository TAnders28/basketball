r"""Pull settled KXNBAGAME markets and 1-minute candles (home YES market). Resumable.

    .venv\Scripts\python.exe -m InPlay.PullKalshi
"""

import argparse
import calendar
import os
import re
import sys
from datetime import datetime

from InPlay.Http import GetJson, HttpError, ReadJson, WriteJsonAtomic

BASE = "https://api.elections.kalshi.com/trade-api/v2"
ROOT = os.path.join("data", "inplay", "kalshi")
EVENT_RE = re.compile(r"^KXNBAGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{2,3})([A-Z]{2,3})$")
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def IsoToTs(text):
    return calendar.timegm(datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").timetuple())


def ParseEventTicker(event_ticker, yes_sub_title=None, ticker=None):
    """KXNBAGAME-26NOV11BOSTOR -> date 2026-11-11, away BOS, home TOR.

    Codes can be 2 or 3 letters, so the split uses the market ticker suffix
    (the YES team) when the concatenation is ambiguous.
    """
    if not event_ticker or not event_ticker.startswith("KXNBAGAME-"):
        return None
    body = event_ticker[len("KXNBAGAME-"):]
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})([A-Z]+)$", body)
    if not m:
        return None
    yy, mon, dd, teams = m.groups()
    if mon not in MONTHS:
        return None
    date = f"20{yy}-{MONTHS[mon]:02d}-{int(dd):02d}"
    side = ticker.rsplit("-", 1)[-1] if ticker else None
    if side and teams.endswith(side):
        away, home = teams[: -len(side)], side
    elif side and teams.startswith(side):
        away, home = side, teams[len(side):]
    elif len(teams) == 6:
        away, home = teams[:3], teams[3:]
    else:
        return None
    return {"date": date, "away": away, "home": home}


def MarketRow(raw, source):
    parsed = ParseEventTicker(raw.get("event_ticker"), ticker=raw.get("ticker"))
    if not parsed:
        return None
    side = raw["ticker"].rsplit("-", 1)[-1]
    return {
        "ticker": raw["ticker"], "event_ticker": raw["event_ticker"], "date_et": parsed["date"],
        "away": parsed["away"], "home": parsed["home"], "side": side, "is_home": side == parsed["home"],
        "result": raw.get("result"), "close_time": raw.get("close_time"),
        "occurrence_datetime": raw.get("occurrence_datetime"), "open_time": raw.get("open_time"),
        "settlement_ts": raw.get("settlement_ts"), "volume": raw.get("volume_fp") or raw.get("volume"),
        "title": raw.get("title"), "source": source,
    }


def Enumerate(get=GetJson):
    rows = {}
    for source, path, extra in (("historical", "/historical/markets", {}),
                                ("live", "/markets", {"status": "settled"})):
        cursor = None
        while True:
            params = {"series_ticker": "KXNBAGAME", "limit": 1000, **extra}
            if cursor:
                params["cursor"] = cursor
            body = get(BASE + path, params=params)
            for raw in body.get("markets", []):
                row = MarketRow(raw, source)
                if row and row["ticker"] not in rows:
                    rows[row["ticker"]] = row
            cursor = body.get("cursor")
            print(f"{source}: {len(rows)} markets", flush=True)
            if not cursor or not body.get("markets"):
                break
    return sorted(rows.values(), key=lambda r: (r["date_et"], r["ticker"]))


def _Num(value):
    if value is None or value == "":
        return None
    return float(value)


def _Pick(block, key):
    if not isinstance(block, dict):
        return None
    if key in block:
        return _Num(block[key])
    return _Num(block.get(key + "_dollars"))


def NormalizeCandle(raw):
    """Both the historical (close) and live (close_dollars) schemas -> flat floats."""
    price = raw.get("price") or {}
    ask = raw.get("yes_ask") or {}
    bid = raw.get("yes_bid") or {}
    return {
        "t": int(raw["end_period_ts"]),
        "bid": _Pick(bid, "close"), "ask": _Pick(ask, "close"),
        "bid_lo": _Pick(bid, "low"), "ask_hi": _Pick(ask, "high"),
        "last": _Pick(price, "close"), "tr_lo": _Pick(price, "low"), "tr_hi": _Pick(price, "high"),
        "vol": _Num(raw.get("volume_fp", raw.get("volume"))) or 0.0,
        "oi": _Num(raw.get("open_interest_fp", raw.get("open_interest"))),
    }


def PullCandles(row, cutoff_ts, get=GetJson, hours_before=5):
    end = IsoToTs(row["close_time"])
    start = end - hours_before * 3600
    settled = IsoToTs(row["settlement_ts"]) if row.get("settlement_ts") else end
    if settled < cutoff_ts:
        url = f"{BASE}/historical/markets/{row['ticker']}/candlesticks"
    else:
        url = f"{BASE}/series/KXNBAGAME/markets/{row['ticker']}/candlesticks"
    body = get(url, params={"start_ts": start, "end_ts": end, "period_interval": 1})
    return [NormalizeCandle(c) for c in body.get("candlesticks", [])]


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-list", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    list_path = os.path.join(ROOT, "markets.json")
    if args.refresh_list or not os.path.exists(list_path):
        WriteJsonAtomic(list_path, Enumerate())
    rows = ReadJson(list_path)
    cutoff = IsoToTs(GetJson(BASE + "/historical/cutoff")["market_settled_ts"])
    home = [r for r in rows if r["is_home"]]
    print(f"{len(rows)} markets, {len(home)} home markets", flush=True)
    done = errors = 0
    for i, row in enumerate(home):
        path = os.path.join(ROOT, "candles", row["ticker"] + ".json")
        if os.path.exists(path):
            continue
        try:
            candles = PullCandles(row, cutoff)
            WriteJsonAtomic(path, {"ticker": row["ticker"], "candles": candles})
            done += 1
        except HttpError as exc:
            errors += 1
            print(f"ERR {row['ticker']} {exc}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(home)} pulled={done} errors={errors}", flush=True)
        if args.limit and done >= args.limit:
            break
    print(f"DONE pulled={done} errors={errors}", flush=True)


if __name__ == "__main__":
    sys.exit(Main())