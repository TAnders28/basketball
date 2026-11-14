"""Pre-registered Kalshi news-latency study vs official NBA injury reports.

Parses downloaded official PDFs, detects status changes, pulls 1-minute
KXNBAGAME candles from -30 to +60 around pre-tip changes, measures reprice
half-life, freezes buy-at-+N on the chronological first half of headline
events, then scores the second half once.

    .venv\\Scripts\\python.exe Backtest/ExportInjuryLatency.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MAIN = Path(r"E:\Basketball\basketball")
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from Backtest.InjuryReports import ParsePdf, TEAM_ABBREV  # noqa: E402

ET = ZoneInfo("America/New_York")
REPORT_DIR = MAIN / "data" / "injury" / "reports"
PARSE_DIR = MAIN / "data" / "injury" / "parsed"
CANDLE_DIR = MAIN / "data" / "inplay" / "kalshi" / "candles_injury"
MARKETS = MAIN / "data" / "inplay" / "kalshi" / "markets.json"
GAMES_META = MAIN / "data" / "inplay" / "games.pkl"
PLAYER_DIR = MAIN / "data" / "matchups"
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_injury-latency")
BASE = "2026-11-13_injury-latency"
FEE = 0.02
FEE_BASIS = "tim_est_0.02"
EDGE_CUT = 0.03
BOOT_REPS = 5000
BOOT_SEED = 20261113
N_GRID = (1, 3, 5, 10)
MIN_TUNE_N = 20
PARSE_WORKERS_MAX = 4
COMBINED_WORKER_CAP = 7


def DetectInPlayWpWorkers() -> int:
    """Count running InPlay.ExportInPlayWp processes (do not touch their files)."""
    try:
        import subprocess
        cmd = "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object -ExpandProperty CommandLine"
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return 0
    return sum(1 for line in out.splitlines() if "ExportInPlayWp" in line)


def ChooseParseWorkers() -> int:
    wp = DetectInPlayWpWorkers()
    # If the WP job is still up it may later spawn up to 7 workers; stay at 1.
    if wp:
        return 1
    return PARSE_WORKERS_MAX

PARSE_WORKERS = ChooseParseWorkers()
WINDOW_BEFORE = 30 * 60
WINDOW_AFTER = 60 * 60
QGTD = {"Questionable", "Doubtful"}
PURE_GLEAGUE_RE = re.compile(r"^gleague", re.I)

EVENTS_FIELDS = [
    "event_id", "split", "season", "game_date", "tip_ts", "report_ts", "minutes_before_tip",
    "player", "team", "opp", "side_bought", "is_starter", "family", "from_status", "to_status",
    "reason", "result",
]
REPRICE_FIELDS = [
    "event_id", "family", "split", "is_starter", "n_post_bars",
    "pre_mid", "settled_mid", "settled_source", "move", "half_life_min", "half_life_censored",
    "ask_plus1", "ask_plus3", "ask_plus5", "ask_plus10",
    "mid_plus1", "mid_plus3", "mid_plus5", "mid_plus10",
]
STRAT_FIELDS = [
    "strategy", "split", "n", "n_days", "wins", "mean_price", "fair_value",
    "mean_edge", "edge_ci95_lo", "edge_ci95_hi", "roi", "roi_ci95_lo", "roi_ci95_hi",
    "mean_clv", "clv_ci95_lo", "clv_ci95_hi", "clears_3c", "note",
]
HALFLIFE_FIELDS = [
    "family", "split", "n_events", "n_with_move", "n_censored",
    "median_half_life_min", "p25_half_life_min", "p75_half_life_min",
    "median_abs_move", "share_halfway_by_1", "share_halfway_by_3",
    "share_halfway_by_5", "share_halfway_by_10",
]


def LimitCpu() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
        )
    except Exception:
        pass


def Log(msg: str) -> None:
    line = f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    path = OUT_DIR / "_export.log"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def GitHash(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), text=True).strip()


def CanonicalHash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def WriteCsv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k)
                if v is None or v == "":
                    out[k] = ""
                elif isinstance(v, bool):
                    out[k] = "True" if v else "False"
                elif isinstance(v, float):
                    if math.isfinite(v) and v == math.trunc(v):
                        out[k] = f"{v:.1f}"
                    else:
                        out[k] = f"{v:.10g}"
                else:
                    out[k] = v
            w.writerow(out)


def AppendTrial(path: Path, rec: dict) -> None:
    rec = dict(rec)
    rec.setdefault("ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def Alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def NameKey(name: str):
    raw = (name or "").strip()
    if "," in raw:
        last, first = raw.split(",", 1)
    else:
        parts = raw.replace("-", " ").split()
        if not parts:
            return "", ""
        first, last = parts[0], " ".join(parts[1:]) if len(parts) > 1 else parts[0]
    return Alnum(last), Alnum(first)


def IsoDate(mmddyyyy: str) -> str:
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})$", mmddyyyy or "")
    if not m:
        return ""
    return f"{int(m.group(3)):04d}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"


def SeasonOf(date: str) -> str:
    return "2025-26" if date <= "2026-06-30" else "2026-27"


def IsPureGLeague(reason: str) -> bool:
    r = re.sub(r"\s+", "", reason or "")
    return bool(PURE_GLEAGUE_RE.match(r))


def Mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def Quantile(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return ys[i]


def BootstrapDayMean(pairs, reps=BOOT_REPS, seed=BOOT_SEED, level=0.95):
    if not pairs:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_days": 0}
    values = [v for _d, v in pairs]
    point = sum(values) / len(values)
    bucket = defaultdict(list)
    for d, v in pairs:
        bucket[d].append(v)
    keys = list(bucket)
    rng = random.Random(seed)
    boots = []
    n_cl = len(keys)
    for _ in range(reps):
        draw = [keys[rng.randrange(n_cl)] for _ in range(n_cl)]
        s = []
        for k in draw:
            s.extend(bucket[k])
        if s:
            boots.append(sum(s) / len(s))
    boots.sort()
    alpha = (1.0 - level) / 2.0
    lo = boots[int(alpha * len(boots))] if boots else None
    hi = boots[min(len(boots) - 1, int((1.0 - alpha) * len(boots)))] if boots else None
    return {"mean": point, "lo": lo, "hi": hi, "n": len(values), "n_days": n_cl}


def BootstrapDayRoi(triples, reps=BOOT_REPS, seed=BOOT_SEED, level=0.95):
    if not triples:
        return {"roi": None, "lo": None, "hi": None, "n": 0, "n_days": 0}
    tot_c = sum(c for _d, c, _p in triples)
    point = (sum(p for _d, _c, p in triples) / tot_c) if tot_c else None
    bucket = defaultdict(list)
    for d, c, p in triples:
        bucket[d].append((c, p))
    keys = list(bucket)
    rng = random.Random(seed)
    boots = []
    n_cl = len(keys)
    for _ in range(reps):
        draw = [keys[rng.randrange(n_cl)] for _ in range(n_cl)]
        sc = sp = 0.0
        for k in draw:
            for c, p in bucket[k]:
                sc += c
                sp += p
        if sc > 0:
            boots.append(sp / sc)
    boots.sort()
    alpha = (1.0 - level) / 2.0
    lo = boots[int(alpha * len(boots))] if boots else None
    hi = boots[min(len(boots) - 1, int((1.0 - alpha) * len(boots)))] if boots else None
    return {"roi": point, "lo": lo, "hi": hi, "n": len(triples), "n_days": n_cl}


def FmtSigned(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}"


def Fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def FmtCI(lo, hi, nd=4):
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def MdTable(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _ParseOne(path_str: str):
    path = Path(path_str)
    cache = PARSE_DIR / (path.stem + ".json")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    rec = ParsePdf(path)
    rec["header_ts"] = rec["header_ts"].isoformat() if rec.get("header_ts") else None
    rec["creation_ts"] = rec["creation_ts"].isoformat() if rec.get("creation_ts") else None
    PARSE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, default=str), encoding="utf-8")
    tmp.replace(cache)
    return rec


def ParseAll(paths):
    PARSE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    n = len(paths)
    Log(f"parsing {n} reports with {PARSE_WORKERS} workers")
    # Windows spawn: worker is _ParseOne in this module
    with ProcessPoolExecutor(max_workers=PARSE_WORKERS) as ex:
        futs = {ex.submit(_ParseOne, str(p)): p for p in paths}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            out.append(rec)
            done += 1
            if done % 200 == 0 or done == n:
                Log(f"parsed {done}/{n}")
    out.sort(key=lambda r: (r.get("header_ts") or "", r.get("name") or ""))
    return out


def DetectChanges(parsed):
    """First report that shows a new status for (game_date, matchup, team, player)."""
    last = {}
    changes = []
    eid = 0
    for report in parsed:
        ts = report.get("header_ts")
        if not ts:
            continue
        for row in report.get("rows") or []:
            key = (row.get("game_date"), row.get("matchup"), row.get("team"), row.get("player"))
            if not all(key):
                continue
            prev = last.get(key)
            if prev is None:
                last[key] = {"status": row["status"], "reason": row.get("reason"), "ts": ts, "first": True}
                continue
            if prev["status"] != row["status"]:
                eid += 1
                changes.append({
                    "change_id": f"chg_{eid:06d}",
                    "report_ts": ts,
                    "game_date_raw": row["game_date"],
                    "game_date": IsoDate(row["game_date"]),
                    "game_time": row.get("game_time"),
                    "matchup": row["matchup"],
                    "team": row["team"],
                    "player": row["player"],
                    "from_status": prev["status"],
                    "to_status": row["status"],
                    "reason": row.get("reason") or "",
                    "prev_reason": prev.get("reason") or "",
                    "report_name": report.get("name"),
                })
            last[key] = {"status": row["status"], "reason": row.get("reason"), "ts": ts, "first": False}
    return changes




def LoadGamesAndMarkets():
    import pickle
    import pandas as pd
    frame = pickle.loads(GAMES_META.read_bytes())
    games = {}
    for rec in frame.itertuples(index=False):
        date = str(rec.date_et)[:10]
        home, away = rec.home, rec.away
        st = None if pd.isna(rec.sched_ts) else int(rec.sched_ts)
        tk = None if pd.isna(rec.kalshi_ticker) else str(rec.kalshi_ticker)
        games[(date, home, away)] = {"date": date, "home": home, "away": away, "sched_ts": st, "ticker": tk}
    markets = json.loads(MARKETS.read_text(encoding="utf-8"))
    for m in markets:
        if not m.get("is_home"):
            continue
        key = (m["date_et"], m["home"], m["away"])
        g = games.get(key)
        if g is None:
            games[key] = {"date": m["date_et"], "home": m["home"], "away": m["away"], "sched_ts": None, "ticker": m.get("ticker")}
        elif not g.get("ticker"):
            g["ticker"] = m.get("ticker")
    results = {}
    for season in ("2025", "2026"):
        path = MAIN / "data" / "season" / f"games_{season}.json"
        if not path.exists():
            continue
        for g in json.loads(path.read_text(encoding="utf-8")):
            date = str(g.get("game_date") or "")[:10]
            home, away = g.get("home_team"), g.get("away_team")
            if date and home and away and g.get("home_win") is not None:
                results[(date, str(home), str(away))] = {
                    "home_win": int(bool(g["home_win"])) if not isinstance(g["home_win"], int) else int(g["home_win"]),
                    "game_id": str(g.get("game_id") or ""),
                }
                # home_win in file is True/False
                results[(date, str(home), str(away))]["home_win"] = 1 if g["home_win"] else 0
    return games, results


def LoadMinutesIndex():
    """team -> list of (date, players)."""
    by_team = defaultdict(list)
    for year in ("2024", "2025", "2026"):
        path = PLAYER_DIR / f"player_games_{year}.json"
        if not path.exists():
            continue
        # group by (team, date)
        tmp = defaultdict(list)
        for r in json.loads(path.read_text(encoding="utf-8")):
            team = r.get("TEAM_ABBREVIATION")
            date = str(r.get("GAME_DATE") or "")[:10]
            try:
                mins = float(r.get("MIN"))
            except (TypeError, ValueError):
                continue
            last, first = NameKey(r.get("PLAYER_NAME") or "")
            tmp[(team, date)].append({"name": r.get("PLAYER_NAME"), "last": last, "first": first, "min": mins})
        for (team, date), plist in tmp.items():
            by_team[team].append((date, plist))
    for team in by_team:
        by_team[team].sort()
    return by_team


def NameMatch(last, first, rec):
    if rec["last"] != last:
        return False
    if not first or not rec["first"]:
        return True
    return rec["first"].startswith(first[:3]) or first.startswith(rec["first"][:3])


def StarterFlag(player, team, game_date, minutes_index):
    last, first = NameKey(player)
    if game_date <= "2026-06-30":
        season_start, season_end = "2025-10-01", "2026-06-30"
        prev_start, prev_end = "2024-10-01", "2025-06-30"
    else:
        season_start, season_end = "2026-10-01", "2027-06-30"
        prev_start, prev_end = "2025-10-01", "2026-06-30"
    team_games = minutes_index.get(team) or []
    priors = [(d, pl) for d, pl in team_games if season_start <= d < game_date and d <= season_end]
    how = "season_top5"
    if len(priors) < 3:
        priors = [(d, pl) for d, pl in team_games if prev_start <= d <= prev_end]
        how = "prev_season_top5"
    if not priors:
        return False, None, 0, "no_minutes"
    sums = defaultdict(lambda: [0.0, 0])
    mine = []
    for d, plist in priors:
        for r in plist:
            key = (r["last"], r["first"])
            sums[key][0] += r["min"]
            sums[key][1] += 1
            if NameMatch(last, first, r):
                mine.append(r["min"])
    means = sorted(((tot / cnt, key) for key, (tot, cnt) in sums.items() if cnt), reverse=True)
    top5 = {key for _m, key in means[:5]}
    is_st = any((k in top5) for k in top5 if k[0] == last and (not first or k[1].startswith(first[:3]) or first.startswith(k[1][:3])))
    # clearer
    is_st = False
    for _m, key in means[:5]:
        if key[0] == last and (not first or key[1].startswith(first[:3]) or first.startswith(key[1][:3])):
            is_st = True
            break
    mpg = (sum(mine) / len(mine)) if mine else None
    return bool(is_st), mpg, len(priors), how


def AttachContext(changes, games, results, minutes_index):
    out = []
    skipped = Counter()
    worse = {"Available": 0, "Probable": 1, "Questionable": 2, "Doubtful": 3, "Out": 4}
    for ch in changes:
        date = ch["game_date"]
        matchup = ch["matchup"]
        if not date or "@" not in (matchup or ""):
            skipped["bad_key"] += 1
            continue
        away, home = matchup.split("@", 1)
        g = games.get((date, home, away))
        if not g or not g.get("sched_ts"):
            skipped["no_sched"] += 1
            continue
        try:
            report_ts = int(datetime.fromisoformat(ch["report_ts"]).timestamp())
        except Exception:
            skipped["bad_ts"] += 1
            continue
        if report_ts >= int(g["sched_ts"]):
            skipped["after_tip"] += 1
            continue
        if not g.get("ticker"):
            skipped["no_ticker"] += 1
            continue
        res = results.get((date, home, away))
        if res is None or res.get("home_win") is None:
            skipped["no_result"] += 1
            continue
        is_starter, mpg, n_prior, how = StarterFlag(ch["player"], ch["team"], date, minutes_index)
        injured_is_home = ch["team"] == home
        to_out = ch["to_status"] == "Out"
        from_q = ch["from_status"] in QGTD
        from_out = ch["from_status"] == "Out"
        to_up = ch["to_status"] in {"Available", "Probable"}
        if to_out:
            side = "away" if injured_is_home else "home"
            family_core = "downgrade_to_out"
        elif from_out and to_up:
            side = "home" if injured_is_home else "away"
            family_core = "upgrade_from_out"
        elif worse.get(ch["to_status"], 0) > worse.get(ch["from_status"], 0):
            side = "away" if injured_is_home else "home"
            family_core = "other_downgrade"
        else:
            side = "home" if injured_is_home else "away"
            family_core = "other_upgrade"
        headline = bool(is_starter and from_q and to_out and not IsPureGLeague(ch["reason"]))
        rec = dict(ch)
        rec.update({
            "home": home, "away": away,
            "sched_ts": int(g["sched_ts"]),
            "ticker": g["ticker"],
            "report_epoch": report_ts,
            "minutes_before_tip": (int(g["sched_ts"]) - report_ts) / 60.0,
            "home_win": int(res["home_win"]),
            "game_id": res.get("game_id") or f"{date}_{away}@{home}",
            "is_starter": bool(is_starter),
            "starter_mpg": mpg,
            "starter_n_prior": n_prior,
            "starter_how": how,
            "side_bought": side,
            "family_core": family_core,
            "headline": headline,
            "pure_gleague": IsPureGLeague(ch["reason"]),
            "season": SeasonOf(date),
        })
        rec["result"] = rec["home_win"] if side == "home" else 1 - rec["home_win"]
        out.append(rec)
    return out, skipped


def SideQuote(candle, side):
    """Home YES candle -> bid/ask/mid for a side."""
    if candle is None:
        return None
    hb, ha = candle.get("bid"), candle.get("ask")
    if hb is None or ha is None:
        return None
    hmid = (float(hb) + float(ha)) / 2.0
    if side == "home":
        return {"bid": float(hb), "ask": float(ha), "mid": hmid, "t": int(candle["t"])}
    return {"bid": 1.0 - float(ha), "ask": 1.0 - float(hb), "mid": 1.0 - hmid, "t": int(candle["t"])}


def LastAtOrBefore(candles, t_end, t_min=None):
    pick = None
    for c in candles:
        tt = int(c["t"])
        if t_min is not None and tt < t_min:
            continue
        if tt <= t_end:
            pick = c
        else:
            break
    return pick


def FirstAtOrAfter(candles, t_start, t_max=None, slack=120):
    for c in candles:
        tt = int(c["t"])
        if tt < t_start:
            continue
        if t_max is not None and tt > t_max:
            return None
        if tt <= t_start + slack:
            return c
        return c if tt - t_start <= 180 else None
    return None


def PullEventCandles(event, cutoff_ts):
    from InPlay.Http import GetJson, HttpError
    from InPlay.PullKalshi import NormalizeCandle, BASE
    ticker = event["ticker"]
    cache = CANDLE_DIR / f"{ticker}_{event['change_id']}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8")).get("candles") or []
    start = int(event["report_epoch"]) - WINDOW_BEFORE
    end = int(event["report_epoch"]) + WINDOW_AFTER
    settled_end = int(event["sched_ts"])
    # choose historical vs live by market settlement vs cutoff: use sched as proxy if unknown
    if settled_end < cutoff_ts:
        url = f"{BASE}/historical/markets/{ticker}/candlesticks"
    else:
        url = f"{BASE}/series/KXNBAGAME/markets/{ticker}/candlesticks"
    try:
        body = GetJson(url, params={"start_ts": start, "end_ts": end, "period_interval": 1})
        candles = [NormalizeCandle(c) for c in body.get("candlesticks", [])]
    except HttpError as exc:
        Log(f"candle err {ticker} {event['change_id']} {exc}")
        candles = []
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"ticker": ticker, "change_id": event["change_id"], "candles": candles}), encoding="utf-8")
    tmp.replace(cache)
    return candles


def MeasureEvent(event, candles):
    t0 = int(event["report_epoch"])
    tip = int(event["sched_ts"])
    side = event["side_bought"]
    pre_c = LastAtOrBefore(candles, t0, t_min=t0 - WINDOW_BEFORE)
    pre = SideQuote(pre_c, side)
    t60 = t0 + WINDOW_AFTER
    settled_source = "plus60"
    settled_c = LastAtOrBefore(candles, min(t60, tip), t_min=t0)
    if t60 > tip:
        settled_source = "last_pre_tip"
    settled = SideQuote(settled_c, side)
    out = {
        "pre": pre, "settled": settled, "settled_source": settled_source,
        "plus": {}, "half_life_min": None, "half_life_censored": None,
        "move": None, "n_post_bars": 0,
    }
    for n in N_GRID:
        c = FirstAtOrAfter(candles, t0 + n * 60, t_max=tip, slack=120)
        out["plus"][n] = SideQuote(c, side)
    if not pre or not settled:
        return out
    move = settled["mid"] - pre["mid"]
    out["move"] = move
    if abs(move) < 0.005:
        out["half_life_min"] = 0.0
        out["half_life_censored"] = False
        return out
    target = pre["mid"] + 0.5 * move
    going_up = move > 0
    post = [c for c in candles if t0 < int(c["t"]) <= min(t60, tip)]
    out["n_post_bars"] = len(post)
    hit = None
    for c in post:
        q = SideQuote(c, side)
        if q is None:
            continue
        if going_up and q["mid"] >= target:
            hit = c
            break
        if (not going_up) and q["mid"] <= target:
            hit = c
            break
    if hit is None:
        last = post[-1] if post else None
        out["half_life_min"] = ((int(last["t"]) - t0) / 60.0) if last else None
        out["half_life_censored"] = True
    else:
        out["half_life_min"] = (int(hit["t"]) - t0) / 60.0
        out["half_life_censored"] = False
    return out


def TradeAt(event, quote):
    if not quote or quote.get("ask") is None:
        return None
    ask = float(quote["ask"])
    if ask <= 0 or ask >= 1:
        return None
    cost = ask + FEE
    pnl = (1.0 if event["result"] else 0.0) - cost
    clv = None
    return {"ask": ask, "cost": cost, "pnl": pnl, "mid": quote.get("mid")}


def SummarizeTrades(trades, strategy, split, note=""):
    if not trades:
        return {
            "strategy": strategy, "split": split, "n": 0, "n_days": 0, "wins": 0,
            "mean_price": None, "fair_value": None, "mean_edge": None,
            "edge_ci95_lo": None, "edge_ci95_hi": None, "roi": None,
            "roi_ci95_lo": None, "roi_ci95_hi": None, "mean_clv": None,
            "clv_ci95_lo": None, "clv_ci95_hi": None, "clears_3c": False, "note": note,
        }
    edge = BootstrapDayMean([(t["date"], t["pnl"]) for t in trades])
    roi = BootstrapDayRoi([(t["date"], t["cost"], t["pnl"]) for t in trades])
    clvs = [(t["date"], t["clv"]) for t in trades if t.get("clv") is not None]
    clv = BootstrapDayMean(clvs) if clvs else {"mean": None, "lo": None, "hi": None}
    clears = bool(edge["mean"] is not None and edge["mean"] >= EDGE_CUT and edge["lo"] is not None and edge["lo"] > 0)
    return {
        "strategy": strategy, "split": split, "n": len(trades), "n_days": edge["n_days"],
        "wins": int(sum(t["result"] for t in trades)),
        "mean_price": Mean([t["ask"] for t in trades]),
        "fair_value": Mean([t["result"] for t in trades]),
        "mean_edge": edge["mean"], "edge_ci95_lo": edge["lo"], "edge_ci95_hi": edge["hi"],
        "roi": roi["roi"], "roi_ci95_lo": roi["lo"], "roi_ci95_hi": roi["hi"],
        "mean_clv": clv["mean"], "clv_ci95_lo": clv["lo"], "clv_ci95_hi": clv["hi"],
        "clears_3c": clears, "note": note,
    }


def HalfLifeTable(events, family, split):
    hls, moves = [], []
    n_censor = 0
    hit = {1: 0, 3: 0, 5: 0, 10: 0}
    n_move = 0
    for e in events:
        m = e.get("meas") or {}
        if m.get("move") is None:
            continue
        if abs(m["move"]) >= 0.005:
            n_move += 1
            if m.get("half_life_min") is not None:
                # for median, use observed times; censored counted separately
                if m.get("half_life_censored"):
                    n_censor += 1
                else:
                    hls.append(m["half_life_min"])
                    for n in (1, 3, 5, 10):
                        if m["half_life_min"] <= n:
                            hit[n] += 1
        moves.append(abs(m["move"]))
    denom = max(n_move - n_censor, 0)  # among uncensored with a move, share that hit by +N; also report vs all moves
    # shares among events with a real move (censored count as not-yet by that N if hl > N)
    n_for_share = 0
    hit_all = {1: 0, 3: 0, 5: 0, 10: 0}
    all_hls = []
    for e in events:
        m = e.get("meas") or {}
        if m.get("move") is None or abs(m["move"]) < 0.005 or m.get("half_life_min") is None:
            continue
        n_for_share += 1
        all_hls.append(m["half_life_min"])
        if not m.get("half_life_censored"):
            for n in (1, 3, 5, 10):
                if m["half_life_min"] <= n:
                    hit_all[n] += 1
    return {
        "family": family, "split": split, "n_events": len(events),
        "n_with_move": n_for_share, "n_censored": n_censor,
        "median_half_life_min": Quantile(hls, 0.5) if hls else Quantile(all_hls, 0.5),
        "p25_half_life_min": Quantile(hls, 0.25) if hls else None,
        "p75_half_life_min": Quantile(hls, 0.75) if hls else None,
        "median_abs_move": Quantile(moves, 0.5),
        "share_halfway_by_1": (hit_all[1] / n_for_share) if n_for_share else None,
        "share_halfway_by_3": (hit_all[3] / n_for_share) if n_for_share else None,
        "share_halfway_by_5": (hit_all[5] / n_for_share) if n_for_share else None,
        "share_halfway_by_10": (hit_all[10] / n_for_share) if n_for_share else None,
    }


def BuildTrades(events, n):
    trades = []
    for e in events:
        q = (e.get("meas") or {}).get("plus", {}).get(n)
        tr = TradeAt(e, q)
        if tr is None:
            continue
        settled = (e.get("meas") or {}).get("settled")
        clv = (settled["mid"] - tr["ask"]) if settled and settled.get("mid") is not None else None
        trades.append({
            "date": e["game_date"],
            "ask": tr["ask"], "cost": tr["cost"], "pnl": tr["pnl"],
            "result": e["result"], "clv": clv, "n": n,
            "change_id": e["change_id"],
        })
    return trades


def Main() -> int:
    LimitCpu()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = OUT_DIR / f"{BASE}_trials.jsonl"
    repo = GitHash(MAIN)
    AppendTrial(trials, {"phase": "start", "repo_head": repo, "prereg": str(OUT_DIR / f"{BASE}_preregistration.md")})

    cov_path = OUT_DIR / "_pull_coverage.json"
    coverage = json.loads(cov_path.read_text(encoding="utf-8")) if cov_path.exists() else {}
    pdfs = sorted(REPORT_DIR.glob("Injury-Report_*.pdf"))
    if len(pdfs) < 100:
        msg = f"BLOCKER: only {len(pdfs)} official injury-report PDFs on disk; download did not finish or timestamped reports are unavailable."
        (OUT_DIR / f"{BASE}_BLOCKER.txt").write_text(msg + "\n", encoding="utf-8")
        Log(msg)
        return 2
    parsed = ParseAll(pdfs)
    parsed = [r for r in parsed if r.get("header_ts") and r.get("n_rows", 0) > 0]
    Log(f"parsed_ok={len(parsed)}")
    changes = DetectChanges(parsed)
    Log(f"raw_status_changes={len(changes)}")

    games, results = LoadGamesAndMarkets()
    minutes_index = LoadMinutesIndex()
    eligible, skipped = AttachContext(changes, games, results, minutes_index)
    Log(f"pre_tip_eligible={len(eligible)} skipped={dict(skipped)}")

    headline_all = [e for e in eligible if e["headline"]]
    Log(f"headline_starter_qgtd_to_out={len(headline_all)}")

    # Candle cutoff
    from InPlay.Http import GetJson
    from InPlay.PullKalshi import IsoToTs, BASE as KBASE
    try:
        cutoff_ts = IsoToTs(GetJson(KBASE + "/historical/cutoff")["market_settled_ts"])
    except Exception as exc:
        Log(f"cutoff lookup failed ({exc}); defaulting to live endpoint")
        cutoff_ts = 0

    # Pull candles for headline events first (the preregistered family). Reuse any
    # already-cached windows for other events, but do not wait on a full 4k-event
    # pull: this run treats PDF coverage as frozen and scores what we can finish.
    CANDLE_DIR.mkdir(parents=True, exist_ok=True)
    cached_keys = {p.stem for p in CANDLE_DIR.glob("*.json")}
    extras = [e for e in eligible if (not e["headline"]) and f"{e['ticker']}_{e['change_id']}" in cached_keys]
    pull_list = list(headline_all) + extras
    Log(f"candle_pull_queue headlines={len(headline_all)} cached_nonheadline={len(extras)} skipped_uncached={len(eligible) - len(pull_list)}")
    n_candles = 0
    n_empty = 0
    for i, e in enumerate(pull_list, start=1):
        candles = PullEventCandles(e, cutoff_ts)
        e["meas"] = MeasureEvent(e, candles)
        if candles:
            n_candles += 1
        else:
            n_empty += 1
        if i % 25 == 0 or i == len(pull_list):
            Log(f"candles {i}/{len(pull_list)} with_bars={n_candles} empty={n_empty}")
    coverage["candle_pull"] = {
        "pre_tip_eligible": len(eligible),
        "headline": len(headline_all),
        "pulled": len(pull_list),
        "with_bars": n_candles,
        "empty": n_empty,
        "skipped_uncached_nonheadline": len(eligible) - len(pull_list),
        "note": "This run pulled 1-minute windows for headline starter Q/GTD->Out events and any non-headline events already cached from the interrupted full pull. Other pre-tip changes were not waited on.",
    }

    if n_candles == 0 and eligible:
        msg = "BLOCKER: 1-minute Kalshi candles were not available around any pre-tip official injury-status change."
        (OUT_DIR / f"{BASE}_BLOCKER.txt").write_text(msg + "\n", encoding="utf-8")
        Log(msg)
        return 2

    # Chronological first half of HEADLINE events (even first, then the rest)
    headline_all.sort(key=lambda e: (e["report_ts"], e["change_id"]))
    mid = (len(headline_all) + 1) // 2
    tune_h = headline_all[:mid]
    hold_h = headline_all[mid:]

    # Fit N on tune only
    tune_grid = []
    for n in N_GRID:
        trades = BuildTrades(tune_h, n)
        summ = SummarizeTrades(trades, f"headline_plus{n}", "tune")
        tune_grid.append({"n": n, "n_trades": summ["n"], "mean_edge": summ["mean_edge"], "roi": summ["roi"], "eligible": summ["n"] >= MIN_TUNE_N, "summ": summ})
        AppendTrial(trials, {"phase": "tune_grid", "n": n, "n_trades": summ["n"], "mean_edge": summ["mean_edge"], "roi": summ["roi"], "eligible": summ["n"] >= MIN_TUNE_N})

    eligible_grid = [g for g in tune_grid if g["eligible"] and g["mean_edge"] is not None]
    if eligible_grid:
        eligible_grid.sort(key=lambda g: (-g["mean_edge"], g["n"]))
        chosen_n = eligible_grid[0]["n"]
        chosen_reason = "max tune mean realised edge among N with >= 20 tune trades; ties prefer shorter N"
    else:
        chosen_n = None
        chosen_reason = "no eligible N with >= 20 tune trades; freeze no bet"

    frozen = {
        "job": "injury-latency",
        "sport": "nba",
        "fee": FEE,
        "fee_basis": FEE_BASIS,
        "headline": "starter Q/GTD -> Out, not pure G League, buy opponent at +N min",
        "n_minutes": chosen_n,
        "tune_headline_events": len(tune_h),
        "holdout_headline_events": len(hold_h),
        "tune_grid": [{"n": g["n"], "n_trades": g["n_trades"], "mean_edge": g["mean_edge"], "roi": g["roi"], "eligible": g["eligible"]} for g in tune_grid],
        "choose": chosen_reason,
        "split": "chronological first half of eligible headline events (even first)",
        "min_tune_n": MIN_TUNE_N,
        "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED, "level": 0.95, "cluster": "day"},
        "repo_head_at_freeze": repo,
        "n_reports_parsed": len(parsed),
        "n_pre_tip_changes": len(eligible),
    }
    frozen_hash = CanonicalHash(frozen)
    frozen["sha256"] = frozen_hash
    (OUT_DIR / f"{BASE}_spec.json").write_text(json.dumps(frozen, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / f"{BASE}_spec.sha256").write_text(frozen_hash + "\n", encoding="utf-8")
    AppendTrial(trials, {"phase": "freeze", "sha256": frozen_hash, "chosen_n": chosen_n, "reason": chosen_reason})

    # Score holdout (and descriptive tune) AFTER the hash is on disk.
    strat_rows = []
    families = {
        "headline_starter_qgtd_out": (lambda e: e["headline"], "starter Q/GTD->Out, buy opponent"),
        "starter_any_to_out": (lambda e: e["is_starter"] and e["to_status"] == "Out" and not e["pure_gleague"], "starter any->Out, buy opponent"),
        "starter_upgrade_from_out": (lambda e: e["is_starter"] and e["family_core"] == "upgrade_from_out" and not e["pure_gleague"], "starter Out->Available/Probable, buy team"),
    }

    def SplitEvents(pool, is_tune):
        ids = {e["change_id"] for e in (tune_h if is_tune else hold_h)}
        # For non-headline families, split by the same chronological midpoint in time as the headline split
        if pool and pool is not headline_all:
            cut = tune_h[-1]["report_ts"] if tune_h else None
            if cut is None:
                return []
            return [e for e in pool if (e["report_ts"] <= cut if is_tune else e["report_ts"] > cut)]
        return [e for e in pool if e["change_id"] in ids]

    for fam_name, (pred, note) in families.items():
        pool = [e for e in eligible if pred(e) and e.get("meas")]
        for split, is_tune in (("tune", True), ("holdout", False)):
            evs = SplitEvents(pool, is_tune)
            ns = list(N_GRID) if fam_name != "headline_starter_qgtd_out" or chosen_n is None else list(N_GRID)
            for n in ns:
                trades = BuildTrades(evs, n)
                flag = " [FROZEN]" if (fam_name == "headline_starter_qgtd_out" and n == chosen_n) else ""
                strat_rows.append(SummarizeTrades(trades, f"{fam_name}_plus{n}", split, note + f"; +{n} min" + flag))

    # Half-life tables
    hl_rows = []
    for fam_name, (pred, _note) in families.items():
        pool = [e for e in eligible if pred(e) and e.get("meas")]
        for split, is_tune in (("tune", True), ("holdout", False), ("all", None)):
            if split == "all":
                evs = pool
            else:
                evs = SplitEvents(pool, is_tune)
            hl_rows.append(HalfLifeTable(evs, fam_name, split))

    # Event / reprice tidy rows
    event_rows, reprice_rows = [], []
    headline_ids = {e["change_id"] for e in headline_all}
    tune_ids = {e["change_id"] for e in tune_h}
    for e in eligible:
        if not e.get("meas"):
            continue
        split = "tune" if e["change_id"] in tune_ids else ("holdout" if e["change_id"] in headline_ids or (tune_h and e["report_ts"] > tune_h[-1]["report_ts"]) else "other")
        if e["headline"]:
            split = "tune" if e["change_id"] in tune_ids else "holdout"
        fam = "headline_starter_qgtd_out" if e["headline"] else e["family_core"]
        event_rows.append({
            "event_id": e["change_id"], "split": split, "season": e["season"],
            "game_date": e["game_date"], "tip_ts": datetime.fromtimestamp(e["sched_ts"], tz=ET).isoformat(timespec="seconds"),
            "report_ts": e["report_ts"], "minutes_before_tip": e["minutes_before_tip"],
            "player": e["player"], "team": e["team"], "opp": e["away"] if e["team"] == e["home"] else e["home"],
            "side_bought": e["side_bought"], "is_starter": e["is_starter"], "family": fam,
            "from_status": e["from_status"], "to_status": e["to_status"], "reason": e["reason"],
            "result": e["result"],
        })
        m = e.get("meas") or {}
        plus = m.get("plus") or {}
        reprice_rows.append({
            "event_id": e["change_id"], "family": fam, "split": split, "is_starter": e["is_starter"],
            "n_post_bars": m.get("n_post_bars"),
            "pre_mid": (m.get("pre") or {}).get("mid"),
            "settled_mid": (m.get("settled") or {}).get("mid"),
            "settled_source": m.get("settled_source"),
            "move": m.get("move"),
            "half_life_min": m.get("half_life_min"),
            "half_life_censored": m.get("half_life_censored"),
            "ask_plus1": (plus.get(1) or {}).get("ask"),
            "ask_plus3": (plus.get(3) or {}).get("ask"),
            "ask_plus5": (plus.get(5) or {}).get("ask"),
            "ask_plus10": (plus.get(10) or {}).get("ask"),
            "mid_plus1": (plus.get(1) or {}).get("mid"),
            "mid_plus3": (plus.get(3) or {}).get("mid"),
            "mid_plus5": (plus.get(5) or {}).get("mid"),
            "mid_plus10": (plus.get(10) or {}).get("mid"),
        })

    WriteCsv(OUT_DIR / f"{BASE}_events.csv", event_rows, EVENTS_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_reprice.csv", reprice_rows, REPRICE_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_strategies.csv", strat_rows, STRAT_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_halflife.csv", hl_rows, HALFLIFE_FIELDS)

    hold_rows = [r for r in strat_rows if r["split"] == "holdout"]
    frozen_hold = next((r for r in hold_rows if r["strategy"] == f"headline_starter_qgtd_out_plus{chosen_n}"), None) if chosen_n else None
    any_clear = bool(frozen_hold and frozen_hold["clears_3c"])
    verdict = (
        "FAIL (the frozen headline rule clears 3c with a 95% day-block CI above 0 on the holdout)"
        if any_clear else
        "PASS (the frozen headline rule does not clear 3c with a 95% day-block CI above 0 on the holdout)"
    )
    if chosen_n is None:
        verdict = "PASS (no eligible N was frozen on the tune half; no holdout bet)"

    for r in strat_rows:
        AppendTrial(trials, {"phase": "score", "strategy": r["strategy"], "split": r["split"], "n": r["n"],
                             "mean_edge": r["mean_edge"], "edge_ci95_lo": r["edge_ci95_lo"], "edge_ci95_hi": r["edge_ci95_hi"],
                             "mean_clv": r["mean_clv"], "clears_3c": r["clears_3c"], "frozen_hash": frozen_hash})

    def StratMd(rows):
        out = []
        for r in rows:
            out.append([
                r["strategy"], str(r["n"]), Fmt(r["mean_price"], 2) if r["n"] else "-",
                Fmt(r["fair_value"], 2) if r["n"] else "-",
                FmtSigned(r["mean_edge"]), FmtCI(r["edge_ci95_lo"], r["edge_ci95_hi"]),
                FmtSigned(r["roi"]), FmtSigned(r["mean_clv"]),
                "yes" if r["clears_3c"] else "no",
            ])
        return out

    def HlMd(rows):
        out = []
        for r in rows:
            out.append([
                r["family"], r["split"], str(r["n_events"]), str(r["n_with_move"]), str(r["n_censored"]),
                Fmt(r["median_half_life_min"], 2), Fmt(r["p25_half_life_min"], 2), Fmt(r["p75_half_life_min"], 2),
                Fmt(r["median_abs_move"], 3),
                Fmt(r["share_halfway_by_1"], 3), Fmt(r["share_halfway_by_3"], 3),
                Fmt(r["share_halfway_by_5"], 3), Fmt(r["share_halfway_by_10"], 3),
            ])
        return out

    frozen_line = "Frozen rule: no eligible N; headline holdout is not bet."
    if chosen_n is not None and frozen_hold:
        frozen_line = (
            f"**Frozen N = +{chosen_n} min.** Tune headline events={len(tune_h)}. "
            f"Holdout N={frozen_hold['n']}, mean edge {FmtSigned(frozen_hold['mean_edge'])} "
            f"{FmtCI(frozen_hold['edge_ci95_lo'], frozen_hold['edge_ci95_hi'])}, "
            f"mean CLV {FmtSigned(frozen_hold['mean_clv'])}."
        )
    elif chosen_n is not None:
        frozen_line = f"**Frozen N = +{chosen_n} min**, but the holdout produced no priced trades."

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    md = f"""# {BASE} - official injury-report news latency on Kalshi (NBA, pre-registered)

Written {now} (ET). Research simulation only; nothing was bet. Pre-registration: `{BASE}_preregistration.md` (written before any latency / edge result). Frozen spec sha256 `{frozen_hash}`.

**Verdict: {verdict}.**

## Coverage of this pull
- The official-PDF pull was frozen mid-run and treated as complete for this scoring. {coverage.get("n_pdfs", len(parsed))} PDFs on disk, dates {coverage.get("date_min")} through {coverage.get("date_max")} ({coverage.get("n_dates")} dates with at least one report). Hourly-era 2025-10-20 through 2025-12-21 is complete (24 slots/day). Most 15-minute dates from 2025-12-22 through 2026-06-13 are complete (96 slots). The All-Star window around 2026-02-13 to 2026-02-17 and most off-season days after the 2025-26 Finals were not on the official host. **2026-27 is only sparsely present (a few probe slots), not a full 15-minute pull.**
- {coverage.get("note", "")}

## Setup
- Official NBA injury report PDFs from official.nba.com (free public). Candle windows: {coverage.get('candle_pull', {})}. Filename form is hourly through 2025-12-21 and 15-minute from 2025-12-22. Event time is the report header timestamp (America/New_York). Parsed {len(parsed)} reports; {len(changes)} raw status changes; {len(eligible)} pre-tip changes with a scheduled tip, Kalshi ticker, and result. Headline starter Q/GTD->Out events: {len(headline_all)} (tune {len(tune_h)}, holdout {len(hold_h)}).
- Cost = ask + $0.02 (Tim's working estimate, unverified, `{FEE_BASIS}`). Headline = mean realised edge per $1 contract with a 95% day-block bootstrap CI ({BOOT_REPS} reps, seed {BOOT_SEED}). 'Clears 3c' = mean >= 0.03 AND CI lower bound > 0.
- N was chosen on the chronological first half of headline events only, then frozen and hashed before the second half was scored. {chosen_reason}.
- **Caveat: official reports publish on a cadence of about 15 minutes (later 2025-26 and 2026-27) or about 1 hour (early 2025-26). News often leaks earlier on social media first.** Away Kalshi prices are implied from the home YES book.

## Reprice speed
{MdTable(["family", "split", "N events", "N with move", "N censored", "median half-life (min)", "p25", "p75", "median |move|", "share halfway by +1", "+3", "+5", "+10"], HlMd(hl_rows))}

## Holdout strategies
{MdTable(["strategy", "N", "mean ask", "fair value", "mean edge", "95% day-block CI", "ROI", "mean CLV", "clears 3c?"], StratMd(hold_rows))}

## Tune (descriptive)
{MdTable(["strategy", "N", "mean ask", "fair value", "mean edge", "95% day-block CI", "ROI", "mean CLV", "clears 3c?"], StratMd([r for r in strat_rows if r["split"]=="tune"]))}

{frozen_line}

## Caveats
- $0.02 is Tim's working estimate, not a verified Kalshi taker fee. Report publication granularity means the true news time may be earlier, and social media often leads the official PDF.
- Only home YES candles are pulled; the away ask is 1 minus the home bid. Queue, size and partial fills are not modelled (taker-at-ask is optimistic relative to a thin book, conservative relative to mid).
- Starter flags use prior-season-to-date team minutes (top 5); early-season players fall back to last season. Name matching is last name + first-three-letter, which can collide.
- Several families and delays are reported; the verdict uses only the pre-registered headline family at the frozen N. Default call is PASS unless that rule's holdout 95% CI lies entirely above 0 and mean edge >= 0.03.
- Skipped pre-tip attachments: {dict(skipped)}. Empty candle windows: {n_empty} of {len(eligible)}.
- Trial registry: `{BASE}_trials.jsonl`. Frozen spec: `{BASE}_spec.json`. Research simulation only; nothing was bet.

Elapsed {time.time() - t0:.1f}s. Repo HEAD at write: {repo}.
"""
    (OUT_DIR / f"{BASE}.md").write_text(md, encoding="utf-8")
    run = {
        "written": now, "repo_head": repo, "frozen_sha256": frozen_hash, "verdict": verdict,
        "chosen_n": chosen_n, "n_reports": len(parsed), "n_changes": len(changes),
        "n_pre_tip": len(eligible), "n_headline": len(headline_all),
        "n_tune": len(tune_h), "n_holdout": len(hold_h),
        "skipped": dict(skipped), "n_empty_candles": n_empty,
        "elapsed_s": time.time() - t0, "pull_coverage": coverage,
    }
    (OUT_DIR / f"{BASE}_run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")
    AppendTrial(trials, {"phase": "done", "verdict": verdict, "sha256": frozen_hash, "elapsed_s": time.time() - t0})
    Log(json.dumps({"ok": True, "verdict": verdict, "sha256": frozen_hash, "chosen_n": chosen_n,
                    "n_headline": len(headline_all), "elapsed_s": round(time.time() - t0, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())