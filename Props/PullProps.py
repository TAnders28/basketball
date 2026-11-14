"""Polite, cached pull of Kalshi NBA player-prop markets, events, and pregame candles."""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from InPlay.Http import GetJson, HttpError, ReadJson, WriteJsonAtomic

BASE = "https://api.elections.kalshi.com/trade-api/v2"
ROOT = Path("data") / "props"
MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

# Core count-prop series to cover. PROT variants are included when present.
TARGET_SERIES = (
    "KXNBA3PT", "KXNBA3PTPROT",
    "KXNBAPTS", "KXNBAPTSPROT",
    "KXNBAREB", "KXNBAREBPROT",
    "KXNBAAST", "KXNBAASTPROT",
    "KXNBAPRA", "KXNBAPRAPROT",
)
# Extra sleep on top of InPlay.Http's 0.25s gap -> about 2.5 req/s.
EXTRA_GAP = 0.15
TITLE_RE = re.compile(r"^(?P<name>.+):\s*(?P<k>\d+)\+?\s*(?P<stat>.*)$", re.I)


def Log(msg):
    print(msg, flush=True)


def Get(path, params=None):
    time.sleep(EXTRA_GAP)
    return GetJson(BASE + path, params=params)


def IsoToTs(text):
    if not text:
        return None
    s = str(text)[:19]
    try:
        return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return None


def ParseEventDate(event_ticker):
    """KXNBA3PT-26NOV13GSWSAS -> date, away, home when the suffix is 6 letters."""
    if not event_ticker or "-" not in event_ticker:
        return None
    body = event_ticker.split("-", 1)[1]
    m = re.match(r"^(\d{2})([A-Z]{3})(\d{2})([A-Z]{2,3})([A-Z]{2,3})$", body)
    if not m:
        return None
    yy, mon, dd, away, home = m.groups()
    if mon not in MONTHS:
        return None
    return {
        "date": f"20{yy}-{MONTHS[mon]:02d}-{int(dd):02d}",
        "away": away,
        "home": home,
    }


def ParseTitle(raw):
    title = str(raw.get("title") or raw.get("yes_sub_title") or "").strip()
    name, k, stat = None, None, title
    m = re.match(r"^(?P<name>.+):\s*(?P<k>\d+)\+\s*(?P<stat>.*)$", title, re.I)
    if not m:
        m = re.match(r"^(?P<name>.+?)\s+records\s+(?P<k>\d+)\+\s*(?P<stat>.*)$", title, re.I)
    if m:
        name = m.group("name").strip()
        k = int(m.group("k"))
        stat = (m.group("stat") or "").strip().lower()
    if k is None:
        ms = re.search(r"-(\d+)$", str(raw.get("ticker") or ""))
        if ms:
            k = int(ms.group(1))
    return {"name": name, "k": k, "stat_text": stat, "title": title}


def StatFamily(series, stat_text):
    s = (series or "").upper()
    t = stat_text or ""
    if "3PT" in s or "three" in t:
        return "3pm"
    if s.endswith("PRA") or "PRA" in s or "pts + reb + ast" in t or "points, rebounds" in t:
        return "pra"
    if "PTS" in s or "point" in t:
        return "pts"
    if "REB" in s or "rebound" in t:
        return "reb"
    if "AST" in s or "assist" in t:
        return "ast"
    return "other"


def ListSeries():
    """Confirm NBA player-prop series via /series (cached)."""
    out = ROOT / "series.json"
    found = []
    cursor = None
    pages = 0
    while pages < 40:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        body = Get("/series", params)
        pages += 1
        rows = body.get("series") or body.get("series_list") or []
        for s in rows:
            ticker = str(s.get("ticker") or s.get("series_ticker") or "")
            title = str(s.get("title") or "")
            blob = f"{ticker} {title} {s.get('category') or ''}".lower()
            if ticker.upper() in TARGET_SERIES or (
                ticker.upper().startswith("KXNBA") and any(
                    k in blob for k in ("player", "three", "points", "rebound", "assist", "pra")
                )
            ):
                found.append({
                    "ticker": ticker,
                    "title": title,
                    "category": s.get("category"),
                    "frequency": s.get("frequency"),
                    "blob": blob,
                })
        cursor = body.get("cursor") or body.get("next_cursor")
        Log(f"series page {pages}: got {len(rows)} keep={len(found)} cursor={bool(cursor)}")
        if not rows or not cursor:
            break
    # Direct probes for the preregistered tickers in case the list is incomplete.
    have = {s["ticker"] for s in found}
    for t in TARGET_SERIES:
        if t in have:
            continue
        try:
            body = Get(f"/series/{t}")
            s = body.get("series") or body
            if isinstance(s, dict) and (s.get("ticker") or s.get("title")):
                found.append({
                    "ticker": s.get("ticker") or t,
                    "title": s.get("title") or "",
                    "category": s.get("category"),
                    "frequency": s.get("frequency"),
                    "blob": f"{t} {s.get('title') or ''}".lower(),
                    "direct": True,
                })
                Log(f"series {t}: {s.get('title')}")
        except HttpError as ex:
            Log(f"series {t}: {ex}")
    WriteJsonAtomic(str(out), found)
    return found


def Paginate(path, params, key="markets", max_pages=80):
    rows, cursor = [], None
    for page in range(max_pages):
        q = dict(params)
        if cursor:
            q["cursor"] = cursor
        try:
            body = Get(path, q)
        except HttpError as ex:
            Log(f"paginate stop {path} {params.get('series_ticker')} {ex}")
            break
        chunk = body.get(key) or body.get("events") or []
        rows.extend(chunk)
        cursor = body.get("cursor")
        Log(f"  {path} {params.get('series_ticker') or ''} {params.get('status') or ''} "
            f"page {page+1}: +{len(chunk)} total={len(rows)} cursor={bool(cursor)}")
        if not chunk or not cursor:
            break
    return rows


def PullCutoff():
    body = Get("/historical/cutoff")
    WriteJsonAtomic(str(ROOT / "cutoff.json"), body)
    Log(f"historical cutoff {body}")
    return body


def MarketRecord(raw, source, series):
    parsed = ParseTitle(raw)
    ev = raw.get("event_ticker") or ""
    evp = ParseEventDate(ev) or {}
    return {
        "ticker": raw.get("ticker"),
        "event_ticker": ev,
        "series": series,
        "source": source,
        "title": parsed["title"],
        "player_name": parsed["name"],
        "k": parsed["k"],
        "stat_text": parsed["stat_text"],
        "family": StatFamily(series, parsed["stat_text"]),
        "date": evp.get("date"),
        "away": evp.get("away"),
        "home": evp.get("home"),
        "result": raw.get("result"),
        "status": raw.get("status"),
        "open_time": raw.get("open_time"),
        "close_time": raw.get("close_time"),
        "occurrence_datetime": raw.get("occurrence_datetime"),
        "expiration_time": raw.get("expiration_time"),
        "expected_expiration_time": raw.get("expected_expiration_time"),
        "settlement_ts": raw.get("settlement_ts") or raw.get("settlement_time"),
        "volume": raw.get("volume_fp") or raw.get("volume"),
        "open_interest": raw.get("open_interest_fp") or raw.get("open_interest"),
        "yes_bid": raw.get("yes_bid_dollars") or raw.get("yes_bid"),
        "yes_ask": raw.get("yes_ask_dollars") or raw.get("yes_ask"),
        "protected": "PROT" in (series or "").upper(),
        "raw_keys": sorted(raw.keys()),
    }


def PullMarkets(series):
    """Settled markets from /historical/markets and live /markets?status=settled."""
    path = ROOT / f"markets_{series}.json"
    rows = {}
    for source, api, extra in (
        ("historical", "/historical/markets", {}),
        ("live_settled", "/markets", {"status": "settled"}),
    ):
        try:
            chunk = Paginate(api, {"series_ticker": series, "limit": 200, **extra}, key="markets")
        except HttpError as ex:
            Log(f"{source} {series} failed: {ex}")
            chunk = []
        for raw in chunk:
            rec = MarketRecord(raw, source, series)
            t = rec.get("ticker")
            if t and t not in rows:
                rows[t] = rec
        Log(f"{series} {source}: unique={len(rows)}")
    out = sorted(rows.values(), key=lambda r: (r.get("date") or "", r.get("ticker") or ""))
    WriteJsonAtomic(str(path), out)
    return out


def PullEvents(series):
    path = ROOT / f"events_{series}.json"
    rows = []
    for status in ("settled", None):
        params = {"series_ticker": series, "limit": 200}
        if status:
            params["status"] = status
        try:
            chunk = Paginate("/events", params, key="events", max_pages=40)
            rows.extend(chunk)
        except HttpError as ex:
            Log(f"events {series} {status}: {ex}")
    # Dedup
    uniq = {}
    for e in rows:
        uniq[e.get("event_ticker") or e.get("ticker") or json_dumps_safe(e)] = e
    out = list(uniq.values())
    WriteJsonAtomic(str(path), out)
    return out


def json_dumps_safe(e):
    import json
    return json.dumps(e, sort_keys=True)[:80]


def _Num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _Pick(block, key):
    if not isinstance(block, dict):
        return None
    if key in block:
        return _Num(block[key])
    return _Num(block.get(key + "_dollars"))


def NormalizeCandle(raw):
    price = raw.get("price") or {}
    ask = raw.get("yes_ask") or {}
    bid = raw.get("yes_bid") or {}
    ts = raw.get("end_period_ts") or raw.get("ts") or raw.get("end_ts")
    return {
        "t": int(ts) if ts is not None else None,
        "bid": _Pick(bid, "close"),
        "ask": _Pick(ask, "close"),
        "bid_size": _Num((bid or {}).get("close_size") or (bid or {}).get("size") or raw.get("yes_bid_size_fp")),
        "ask_size": _Num((ask or {}).get("close_size") or (ask or {}).get("size") or raw.get("yes_ask_size_fp")),
        "last": _Pick(price, "close"),
        "vol": _Num(raw.get("volume_fp", raw.get("volume"))) or 0.0,
        "oi": _Num(raw.get("open_interest_fp", raw.get("open_interest"))),
        "period_interval": raw.get("period_interval"),
    }


def TipTs(row):
    for key in ("occurrence_datetime", "expected_expiration_time", "close_time"):
        ts = IsoToTs(row.get(key))
        if ts:
            return ts, key
    return None, None


def PullCandles(row, cutoff_ts, hours_before=6, hours_after=1):
    """Try historical first if settled before the cutoff, else the live series route.

    period_interval=1 (1 minute), then 60 (hourly) if empty or rejected.
    Window is around tip (or close_time if tip is missing), from hours_before to hours_after.
    """
    tip, tip_src = TipTs(row)
    if tip is None:
        return [], "no_tip_time", None
    start, end = tip - hours_before * 3600, tip + hours_after * 3600
    settled_ts = IsoToTs(row.get("settlement_ts")) or IsoToTs(row.get("close_time")) or tip
    paths = []
    if cutoff_ts and settled_ts < cutoff_ts:
        paths.append(("/historical/markets/" + row["ticker"] + "/candlesticks", "historical"))
        paths.append((f"/series/{row['series']}/markets/{row['ticker']}/candlesticks", "series"))
    else:
        paths.append((f"/series/{row['series']}/markets/{row['ticker']}/candlesticks", "series"))
        paths.append(("/historical/markets/" + row["ticker"] + "/candlesticks", "historical"))
    last_err = None
    for path, src in paths:
        for interval in (1, 60):
            try:
                body = Get(path, {"start_ts": start, "end_ts": end, "period_interval": interval})
                raw = body.get("candlesticks") or body.get("candles") or []
                candles = [NormalizeCandle(c) for c in raw]
                candles = [c for c in candles if c.get("t") is not None]
                if candles:
                    return candles, f"{src}_int{interval}", tip_src
                last_err = f"{src}_int{interval}_empty"
            except HttpError as ex:
                last_err = f"{src}_int{interval}:{ex}"
                # 404/400: try the next path/interval. 429 is already retried in GetJson.
                if "status:429" in str(ex) or "gave_up" in str(ex):
                    raise
                continue
    return [], last_err or "no_candles", tip_src


def QuoteAtT60(candles, tip_ts, stale_s=3600):
    """Last candle at or before tip-60 minutes, not more than stale_s older than that."""
    if not candles or not tip_ts:
        return None
    target = tip_ts - 3600
    eligible = [c for c in candles if c["t"] is not None and c["t"] <= target]
    if not eligible:
        return None
    eligible.sort(key=lambda c: c["t"])
    c = eligible[-1]
    if target - c["t"] > stale_s:
        return None
    ask, bid = c.get("ask"), c.get("bid")
    mid = None
    if ask is not None and bid is not None:
        mid = 0.5 * (float(ask) + float(bid))
    elif ask is not None:
        mid = float(ask)
    elif bid is not None:
        mid = float(bid)
    elif c.get("last") is not None:
        mid = float(c["last"])
    return {
        "t": c["t"],
        "ask": None if ask is None else float(ask),
        "bid": None if bid is None else float(bid),
        "mid": mid,
        "ask_size": c.get("ask_size"),
        "bid_size": c.get("bid_size"),
        "last": c.get("last"),
        "oi": c.get("oi"),
        "vol": c.get("vol"),
        "age_s": target - c["t"],
        "spread": None if (ask is None or bid is None) else float(ask) - float(bid),
    }


def ListOnly():
    ROOT.mkdir(parents=True, exist_ok=True)
    cutoff = PullCutoff()
    series = ListSeries()
    targets = [s["ticker"] for s in series if s["ticker"] in TARGET_SERIES]
    # Keep preregistered order, plus any confirmed extras later in the spec writer.
    summary = []
    for t in TARGET_SERIES:
        if t not in {s["ticker"] for s in series}:
            Log(f"skip missing series {t}")
            continue
        ev = PullEvents(t)
        mk = PullMarkets(t)
        dates = sorted({r["date"] for r in mk if r.get("date")})
        with_result = sum(1 for r in mk if r.get("result") in ("yes", "no", "Yes", "No"))
        summary.append({
            "series": t,
            "n_events": len(ev),
            "n_markets": len(mk),
            "n_with_result": with_result,
            "date_min": dates[0] if dates else None,
            "date_max": dates[-1] if dates else None,
        })
        Log(f"SUMMARY {t}: markets={len(mk)} results={with_result} dates={dates[0] if dates else None}..{dates[-1] if dates else None} events={len(ev)}")
    WriteJsonAtomic(str(ROOT / "list_summary.json"), {"cutoff": cutoff, "series": summary, "listed_at": datetime.now(timezone.utc).isoformat()})
    return {"cutoff": cutoff, "series": summary}


def PullAllCandles(limit_per_series=0):
    cutoff_body = ReadJson(str(ROOT / "cutoff.json")) if (ROOT / "cutoff.json").exists() else PullCutoff()
    cutoff_ts = IsoToTs(cutoff_body.get("market_settled_ts")) if cutoff_body else None
    Log(f"cutoff_ts={cutoff_ts}")
    candle_dir = ROOT / "candles"
    candle_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for series in TARGET_SERIES:
        mpath = ROOT / f"markets_{series}.json"
        if not mpath.exists():
            continue
        markets = ReadJson(str(mpath))
        n = pulled = cached = empty = errors = 0
        for row in markets:
            if not row.get("ticker"):
                continue
            # Fetch only what the backtest needs: settled results in the eval window.
            if row.get("result") not in ("yes", "no", "Yes", "No"):
                continue
            if row.get("date") and row["date"] < "2025-04-14":
                continue
            n += 1
            path = candle_dir / (row["ticker"] + ".json")
            if path.exists():
                cached += 1
                continue
            try:
                candles, src, tip_src = PullCandles(row, cutoff_ts)
                WriteJsonAtomic(str(path), {
                    "ticker": row["ticker"], "series": series, "source": src,
                    "tip_src": tip_src, "n": len(candles), "candles": candles,
                })
                if candles:
                    pulled += 1
                else:
                    empty += 1
            except HttpError as ex:
                errors += 1
                Log(f"ERR {row['ticker']} {ex}")
            if (pulled + empty + errors) % 25 == 0:
                Log(f"{series}: need={n} pulled={pulled} empty={empty} cached={cached} err={errors}")
            if limit_per_series and (pulled + empty) >= limit_per_series:
                break
        stats[series] = {"need": n, "pulled": pulled, "empty": empty, "cached": cached, "errors": errors}
        Log(f"CANDLES {series} {stats[series]}")
    WriteJsonAtomic(str(ROOT / "candle_summary.json"), stats)
    return stats


def Main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--list-only", action="store_true")
    p.add_argument("--candles", action="store_true")
    p.add_argument("--limit-per-series", type=int, default=0)
    args = p.parse_args(argv)
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.list_only or not args.candles:
        ListOnly()
        if args.list_only:
            return 0
    if args.candles:
        PullAllCandles(limit_per_series=args.limit_per_series)
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
