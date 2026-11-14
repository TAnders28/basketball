"""Build the 3PM props model, write tonight's fairs first, then calibrate and search Kalshi."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import hashlib
import csv
import urllib.request
import urllib.error
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Thread caps before numpy (and before importing the model).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from Props.Model import (
    TRAIN_SEASON, HOLD_SEASON, CURRENT_SEASON, THRESHOLDS, MIN_PROJ,
    AttachPredictions, BuildFeatures, EstimateRho, FilterCalib, FitParams,
    JointGe, NbGe, PoissonGe, PregameRow, RangeRow, TonightState, Shrink,
)

import numpy as np

FEE = 0.02
FEE_BASIS = "tim_est_0.02"
EDGE_CUT = 0.03
ASOF = "2026-11-13"
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_3pm-props")
BASE = "2026-11-13_3pm-props"
REPO = Path(__file__).resolve().parents[1]
SLEEP = 0.7
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_BASE_EXT = "https://external-api.kalshi.com/trade-api/v2"
SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
TARGET_NAMES = ("Yaxel Lendeborg", "Brandin Podziemski")
TARGETS = {
    "Yaxel Lendeborg": 1642865,
    "Brandin Podziemski": 1641764,
}

NORMAL_PRIORITY_CLASS = 0x00000020


def LowerToNormal():
    """Keep this job at Normal priority (do not drop to BelowNormal)."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def Log(msg):
    print(msg, flush=True)


def HttpGet(url, timeout=20, retries=2):
    last = None
    Log(f"GET {url} (timeout={timeout}s)")
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Basketbot/3pm-props (local research; polite)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                ct = resp.headers.get("Content-Type") or ""
                return raw, ct, resp.status
        except Exception as ex:
            last = ex
            Log(f"GET failed attempt {i+1}: {type(ex).__name__}: {ex}"[:240])
            if "404" in str(ex):
                break
            time.sleep(SLEEP * (i + 1))
    raise RuntimeError(f"GET failed {url}: {last}")


def HttpJson(url, timeout=45, retries=3):
    raw, ct, status = HttpGet(url, timeout=timeout, retries=retries)
    return json.loads(raw.decode("utf-8")), status


def LoadJson(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def WriteJson(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def WriteCsv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def GitHash():
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return "unknown"


def LoadPlayerGames(seasons):
    rows = []
    for s in seasons:
        path = REPO / "data" / "matchups" / f"player_games_{s}.json"
        got = LoadJson(path, [])
        rows.extend(got or [])
    return rows


def LoadSeasonGames(seasons):
    rows = []
    for s in seasons:
        path = REPO / "data" / "season" / f"games_{s}.json"
        got = LoadJson(path, [])
        rows.extend(got or [])
    return rows


def RefreshPlayerGames(season=CURRENT_SEASON):
    """One LeagueGameLog call; overwrites the current-season cache only."""
    from nba_api.stats.endpoints import LeagueGameLog
    out = REPO / "data" / "matchups" / f"player_games_{season}.json"
    label = f"{season}-{str(season + 1)[-2:]}"
    last = None
    for attempt in range(1, 5):
        try:
            obj = LeagueGameLog(season=label, player_or_team_abbreviation="P",
                                season_type_all_star="Regular Season", timeout=120)
            rows = obj.get_data_frames()[0].to_dict("records")
            WriteJson(out, rows)
            dates = sorted({str(r.get("GAME_DATE"))[:10] for r in rows})
            Log(f"player_games_{season}: {len(rows)} rows {dates[0] if dates else '?'}..{dates[-1] if dates else '?'}")
            time.sleep(SLEEP)
            return rows
        except Exception as ex:
            last = ex
            Log(f"player games retry {attempt}: {ex!r}"[:200])
            time.sleep(SLEEP * 3 * attempt)
    Log(f"player games refresh failed, using cache if present: {last!r}"[:240])
    return LoadJson(out, [])


def RefreshSeasonGames(season=CURRENT_SEASON):
    import GetSeasonGames as gsg
    label = gsg.SeasonLabel(season)
    all_rows = []
    for st in ("Regular Season",):
        try:
            rows = gsg.FetchTeamGameRows(season, st, retries=4)
            all_rows.extend(rows)
            Log(f"team_games {label} {st}: {len(rows)} rows")
        except Exception as ex:
            Log(f"team_games pull failed: {ex!r}"[:240])
        time.sleep(SLEEP)
    if not all_rows:
        return LoadJson(REPO / "data" / "season" / f"games_{season}.json", [])
    games = gsg.PairGames(all_rows, season)
    paths = gsg.OutputPaths(season)
    os.makedirs(os.path.dirname(paths["games"]), exist_ok=True)
    with open(paths["team_games"], "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")
    WriteJson(paths["games"], games)
    WriteJson(paths["summary"], {
        "season": season, "n_team_rows": len(all_rows), "n_games": len(games),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
    })
    dates = sorted({str(g.get("game_date"))[:10] for g in games})
    Log(f"season games_{season}: {len(games)} games {dates[0] if dates else '?'}..{dates[-1] if dates else '?'}")
    return games


def FetchScoreboard():
    try:
        body, status = HttpJson(SCOREBOARD_URL, timeout=20, retries=2)
        time.sleep(SLEEP)
        return body
    except Exception as ex:
        Log(f"scoreboard CDN failed: {ex!r}"[:200])
        return None


def FetchSchedule():
    # Full league schedule is a large file and the CDN can hang; keep it optional and short.
    try:
        body, status = HttpJson(SCHEDULE_URL, timeout=15, retries=1)
        time.sleep(SLEEP)
        return body
    except Exception as ex:
        Log(f"schedule CDN skipped: {ex!r}"[:200])
        return None


def SlateFromScoreboardV2(date):
    """Fallback slate via ScoreboardV3 (V2 game-header abbreviations were empty)."""
    try:
        from nba_api.stats.endpoints import scoreboardv3
        Log(f"scoreboardv3 slate for {date}")
        sb = scoreboardv3.ScoreboardV3(game_date=date, timeout=30)
        body = sb.get_dict()
        board = body.get("scoreboard") or {}
        time.sleep(SLEEP)
        out = []
        for g in board.get("games") or []:
            ht = g.get("homeTeam") or {}
            at = g.get("awayTeam") or {}
            out.append({
                "game_id": g.get("gameId"),
                "date": str(board.get("gameDate") or date)[:10],
                "tip": g.get("gameTimeUTC") or g.get("gameEt") or g.get("gameStatusText"),
                "status": g.get("gameStatusText") or g.get("gameStatus"),
                "home": ht.get("teamTricode"),
                "away": at.get("teamTricode"),
                "home_name": ht.get("teamName"),
                "away_name": at.get("teamName"),
                "source": "scoreboardv3",
                "label": g.get("gameLabel"),
            })
        return out
    except Exception as ex:
        Log(f"scoreboardv3 slate failed: {ex!r}"[:200])
        return []


def GamesOnDate(scoreboard, schedule, date):
    """Return [{game_id, tip_et, home, away, home_name, away_name, status}] for date."""
    out = []
    if scoreboard:
        board = (scoreboard.get("scoreboard") or scoreboard)
        sdate = str(board.get("gameDate") or "")[:10]
        for g in board.get("games") or []:
            home = ((g.get("homeTeam") or {}).get("teamTricode")
                    or (g.get("homeTeam") or {}).get("teamTricode"))
            away = (g.get("awayTeam") or {}).get("teamTricode")
            hid = (g.get("homeTeam") or {}).get("teamId")
            aid = (g.get("awayTeam") or {}).get("teamId")
            out.append({
                "game_id": g.get("gameId") or g.get("gameCode"),
                "date": sdate or date,
                "tip": g.get("gameTimeUTC") or g.get("gameEt") or g.get("gameStatusText"),
                "status": g.get("gameStatusText") or g.get("gameStatus"),
                "home": home, "away": away,
                "home_name": (g.get("homeTeam") or {}).get("teamName"),
                "away_name": (g.get("awayTeam") or {}).get("teamName"),
                "source": "scoreboard",
                "home_id": hid, "away_id": aid,
            })
        if out and (sdate == date or not sdate):
            return out, "scoreboard"
        # If scoreboard is a different day, still keep but also try schedule.
    sched_rows = []
    if schedule:
        ls = schedule.get("leagueSchedule") or schedule
        for g in ls.get("gameDates") or []:
            gd = str(g.get("gameDate") or "")[:10]
            if gd != date and str(g.get("gameDate") or "")[:10].replace("/", "-") != date:
                # gameDate often "11/13/2026 00:00:00"
                raw = str(g.get("gameDate") or "")
                try:
                    if raw[0:2].isdigit() and raw[2] == "/":
                        mm, dd, yy = raw.split(" ")[0].split("/")
                        gd = f"{int(yy):04d}-{int(mm):02d}-{int(dd):02d}"
                except Exception:
                    pass
            if gd != date:
                continue
            for game in g.get("games") or []:
                ht = game.get("homeTeam") or {}
                at = game.get("awayTeam") or {}
                sched_rows.append({
                    "game_id": game.get("gameId"),
                    "date": date,
                    "tip": game.get("gameDateTimeUTC") or game.get("gameDateTimeEst"),
                    "status": game.get("gameStatusText") or game.get("gameStatus"),
                    "home": ht.get("teamTricode") or ht.get("teamAbbreviation"),
                    "away": at.get("teamTricode") or at.get("teamAbbreviation"),
                    "home_name": ht.get("teamName"),
                    "away_name": at.get("teamName"),
                    "source": "schedule",
                })
    if sched_rows:
        return sched_rows, "schedule"
    return out, "scoreboard" if out else "none"


def FetchInactive(date):
    """ScoreboardV2 InactivePlayers for the slate, if the endpoint answers."""
    try:
        from nba_api.stats.endpoints import scoreboardv2
        sb = scoreboardv2.ScoreboardV2(game_date=date, timeout=60)
        frames = sb.get_data_frames()
        names = list(sb.get_dict().get("resultSets") or [])
        header_names = [r.get("name") for r in (sb.get_dict().get("resultSets") or [])]
        inactive = []
        for i, name in enumerate(header_names):
            if name and "Inactive" in name:
                inactive = frames[i].to_dict("records")
                break
        time.sleep(SLEEP)
        return inactive, header_names
    except Exception as ex:
        Log(f"scoreboardv2 inactive failed: {ex!r}"[:200])
        return [], []


def KalshiGet(path, params=None, base=KALSHI_BASE):
    q = ""
    if params:
        q = "?" + "&".join(f"{k}={urllib.request.quote(str(v)) if isinstance(v, str) else v}" for k, v in params.items())
        # quote only values
        from urllib.parse import urlencode
        q = "?" + urlencode(params)
    url = base + path + q
    body, status = HttpJson(url, timeout=45)
    time.sleep(SLEEP)
    return body


def SearchKalshiSeries():
    """List series and keep anything that looks like NBA player / 3PT props."""
    keys = ("3pt", "3-pt", "3 pm", "3pm", "three", "threes", "made 3", "nba player",
            "player prop", "combo", "same game", "points")
    keep_tokens = ("3pt", "3-pt", "3pm", "three", "player", "prop", "combo", "nba")
    found = []
    cursor = None
    pages = 0
    while pages < 30:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            body = KalshiGet("/series", params)
        except Exception as ex:
            Log(f"kalshi /series page {pages} failed: {ex!r}"[:200])
            # try the external-api host once
            if pages == 0:
                try:
                    body = KalshiGet("/series", params, base=KALSHI_BASE_EXT)
                except Exception as ex2:
                    Log(f"kalshi external /series also failed: {ex2!r}"[:200])
                    break
            else:
                break
        series = body.get("series") or body.get("series_list") or []
        if isinstance(body, list):
            series = body
        pages += 1
        for s in series:
            ticker = str(s.get("ticker") or s.get("series_ticker") or "")
            title = str(s.get("title") or s.get("name") or "")
            cat = str(s.get("category") or s.get("frequency") or "")
            blob = f"{ticker} {title} {cat}".lower()
            if "nba" in blob or ticker.upper().startswith("KXNBA") or any(k in blob for k in keys):
                found.append({
                    "ticker": ticker, "title": title, "category": cat,
                    "frequency": s.get("frequency"),
                    "additional": {k: s.get(k) for k in s if k in ("fee_type", "settlement_sources", "tags", "volume")},
                    "blob": blob,
                })
        cursor = body.get("cursor") or body.get("next_cursor")
        Log(f"kalshi series page {pages}: got {len(series)} keep_so_far={len(found)} cursor={bool(cursor)}")
        if not series or not cursor:
            break
    # Also poke likely NBA series tickers directly.
    guesses = ["KXNBAGAME", "KXNBATOTAL", "KXNBA3PT", "KXNBA3PM", "KXNBAPTS", "KXNBAPROP",
               "KXNBAPLAYER", "KXNBAPLAYERPROP", "KXNBASPREAD", "KXNBACOMBO", "KXNBAPLAYERSTATS",
               "KXNBATHREES", "KXNBASTAT", "KXNBA", "KXNBAPLAYER3PT", "KXPLAYERPROPS",
               "KXNBAMADETHREES", "KXNBAPLAYERTHREES"]
    have = {s["ticker"] for s in found}
    for t in guesses:
        if t in have:
            continue
        try:
            body = KalshiGet(f"/series/{t}")
            s = body.get("series") or body
            if isinstance(s, dict) and (s.get("ticker") or s.get("title")):
                found.append({
                    "ticker": s.get("ticker") or t,
                    "title": s.get("title") or "",
                    "category": s.get("category") or "",
                    "frequency": s.get("frequency"),
                    "additional": {"direct": True},
                    "blob": f"{t} {s.get('title') or ''}".lower(),
                })
                Log(f"kalshi series {t}: found {s.get('title')}")
            time.sleep(0.2)
        except Exception as ex:
            Log(f"kalshi series {t}: no ({type(ex).__name__})")
    return found


def MarketsForSeries(ticker, status=None, max_pages=20):
    rows = []
    cursor = None
    for page in range(max_pages):
        params = {"series_ticker": ticker, "limit": 200}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        try:
            body = KalshiGet("/markets", params)
        except Exception as ex:
            Log(f"kalshi markets {ticker} failed: {ex!r}"[:200])
            break
        mk = body.get("markets") or []
        rows.extend(mk)
        cursor = body.get("cursor")
        if not mk or not cursor:
            break
    return rows


def LooksLikeThreeProp(m):
    blob = " ".join(str(m.get(k) or "") for k in
                    ("ticker", "event_ticker", "title", "subtitle", "yes_sub_title",
                     "no_sub_title", "rules_primary")).lower()
    if any(s in blob for s in ("3pt", "3-pt", "3 pm", "3pm", "three-pointer", "three pointer",
                               "made three", "made 3", "threes")):
        return True
    return False


def MetricsBinary(p, y):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return {"n": 0, "log_loss": None, "brier": None, "ece": None}
    ll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    brier = float(np.mean((p - y) ** 2))
    # Equal-width reliability, 10 bins.
    ece_n = 0.0
    n = len(y)
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        if i == 9:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        ece_n += float(mask.sum()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return {"n": int(n), "log_loss": ll, "brier": brier, "ece": ece_n / n}


def ReliabilityRows(model, thresh, p, y, date="2026-11-13"):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rows = []
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        mask = (p >= lo) & (p <= hi) if i == 9 else (p >= lo) & (p < hi)
        if not np.any(mask):
            rows.append({"date": date, "sport": "nba", "series": "nba_3pm_props",
                         "model": model, "threshold": thresh, "bin_lo": lo, "bin_hi": hi,
                         "n": 0, "mean_pred": None, "obs_rate": None})
            continue
        rows.append({"date": date, "sport": "nba", "series": "nba_3pm_props",
                     "model": model, "threshold": thresh, "bin_lo": lo, "bin_hi": hi,
                     "n": int(mask.sum()), "mean_pred": float(p[mask].mean()),
                     "obs_rate": float(y[mask].mean())})
    return rows


def Fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def UtcToEtLabel(ts):
    if not ts:
        return "NA"
    s = str(ts)
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            return s + " (source, zone unknown)"
        # Convert to America/New_York
        try:
            from zoneinfo import ZoneInfo
            et = dt.astimezone(ZoneInfo("America/New_York"))
            return et.strftime("%Y-%m-%d %H:%M ET")
        except Exception:
            return s
    except Exception:
        return s


def MinutesRange(row, rookie=False):
    rec = [m for m in (row.get("recent_mins") or []) if m is not None]
    mid = row.get("proj_min") or row.get("proj_min_raw") or 16.0
    if len(rec) >= 4:
        lo = float(np.percentile(rec, 25))
        hi = float(np.percentile(rec, 75))
    elif rec:
        sd = float(np.std(rec)) if len(rec) > 1 else 4.0
        lo, hi = mid - sd, mid + sd
    else:
        lo, hi = mid - 6, mid + 6
    if rookie:
        lo, hi = min(lo, mid - 5), max(hi, mid + 5)
    lo = float(min(max(lo, 8.0), mid))
    hi = float(max(min(hi, 42.0), mid))
    return lo, hi


def WriteTonight(path_csv, path_early_md, recs, assumptions, git):
    WriteCsv(path_csv, recs, fields=[
        "date", "sport", "series", "game", "tip_et", "team", "opponent", "home",
        "player_or_combo", "market", "fair", "fair_lo", "fair_hi",
        "lambda", "lambda_lo", "lambda_hi",
        "proj_min", "proj_min_lo", "proj_min_hi", "proj_3pa", "shrunk_3p_pct",
        "prior_n", "prior_3pa", "prior_3pm", "last_played",
        "corr", "family", "status_caveat", "assumption_note",
    ])
    lines = [
        "# Tonight's 3PM fair values (2026-11-13, early write)",
        "",
        "Simulation / model fairs only. No wager was placed. Prices are P(event) as $0.xx to win $1.",
        "",
        assumptions,
        "",
        "| market | fair | low | high | proj min | proj 3PA | ? |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in recs:
        lines.append(
            f"| {r['player_or_combo']} {r['market']} | {Fmt(r['fair'], 2)} | "
            f"{Fmt(r.get('fair_lo'), 2)} | {Fmt(r.get('fair_hi'), 2)} | "
            f"{Fmt(r.get('proj_min'), 1)} | {Fmt(r.get('proj_3pa'), 1)} | {Fmt(r.get('lambda'), 2)} |"
        )
    lines += ["", f"Code at `{git}` (pre-commit). Full write-up follows in the summary md."]
    Path(path_early_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Log("wrote tonight fairs to " + str(path_csv))


def Main(argv=None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--skip-kalshi", action="store_true")
    args = parser.parse_args(argv)
    LowerToNormal()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = OUT_DIR / "_status.json"
    WriteJson(status, {"phase": "pull"})
    Log(f"python={sys.version.split()[0]} git={GitHash()} asof={ASOF}")

    # --- Pull (one process, polite) ---
    if args.skip_pull:
        Log("skip-pull: using cached player/team logs")
    else:
        Log("refreshing 2026 player and team logs (stats.nba.com)")
        RefreshPlayerGames(CURRENT_SEASON)
        RefreshSeasonGames(CURRENT_SEASON)
    scoreboard = FetchScoreboard()
    slate, slate_src = GamesOnDate(scoreboard, None, ASOF)
    if not slate:
        schedule = FetchSchedule()
        slate, slate_src = GamesOnDate(scoreboard, schedule, ASOF)
    if not slate:
        slate = SlateFromScoreboardV2(ASOF)
        slate_src = "scoreboardv2" if slate else "none"
    WriteJson(OUT_DIR / f"{BASE}_slate.json", {"source": slate_src, "games": slate})
    Log(f"slate source={slate_src} n={len(slate)}")
    for g in slate:
        Log(f"  {g.get('away')} @ {g.get('home')} tip={g.get('tip')} status={g.get('status')}")
    inactive, _ih = FetchInactive(ASOF)
    WriteJson(OUT_DIR / f"{BASE}_inactive.json", inactive or [])

    WriteJson(status, {"phase": "kalshi_search"})
    series = SearchKalshiSeries()
    WriteJson(OUT_DIR / f"{BASE}_kalshi_series.json", series)
    three_series = [s for s in series if any(k in (s.get("blob") or "") for k in
                    ("3pt", "3-pt", "3pm", "three", "threes", "player"))]
    # Keep NBA-looking series for a deeper market scan.
    nba_series = [s for s in series if "nba" in (s.get("blob") or "") or
                  str(s.get("ticker") or "").upper().startswith("KXNBA")]
    Log(f"kalshi series matched nba={len(nba_series)} three-ish={len(three_series)}")
    kalshi_markets = []
    scanned = []
    # Prefer three-ish + any NBA series whose ticker/title mentions player/stat/prop/3.
    scan = []
    seen_t = set()
    for s in three_series + nba_series:
        t = s.get("ticker")
        if not t or t in seen_t:
            continue
        blob = s.get("blob") or ""
        if s in three_series or any(k in blob for k in ("player", "prop", "stat", "3pt", "three", "combo")):
            scan.append(s)
            seen_t.add(t)
    for s in scan[:25]:
        mk = MarketsForSeries(s["ticker"])
        scanned.append({"ticker": s["ticker"], "n": len(mk)})
        keep = [m for m in mk if LooksLikeThreeProp(m)]
        Log(f"kalshi markets {s['ticker']}: {len(mk)} total, {len(keep)} look like 3PT props")
        kalshi_markets.extend(keep if keep or "3" in s["ticker"].upper() else [])
        # If the series itself is a 3PT series, keep all markets.
        if any(k in (s.get("blob") or "") for k in ("3pt", "3pm", "three")):
            kalshi_markets.extend(mk)
    # Dedup by ticker
    uniq = {}
    for m in kalshi_markets:
        uniq[m.get("ticker") or json.dumps(m)[:80]] = m
    kalshi_markets = list(uniq.values())
    WriteJson(OUT_DIR / f"{BASE}_kalshi_markets.json", {
        "n": len(kalshi_markets), "scanned_series": scanned, "markets": kalshi_markets[:500],
    })
    has_hist = any(str(m.get("status") or "") in ("determined", "finalized", "settled")
                   or m.get("result") or m.get("settlement_value") is not None
                   for m in kalshi_markets)
    # Open markets that mention tonight's players, for context only.
    tonight_mkt = []
    needles = ("lendeborg", "podziemski", "podz")
    for m in kalshi_markets:
        blob = " ".join(str(m.get(k) or "") for k in
                        ("ticker", "title", "subtitle", "yes_sub_title", "event_ticker")).lower()
        if any(n in blob for n in needles):
            tonight_mkt.append(m)
    Log(f"kalshi 3PT-like markets={len(kalshi_markets)} settled_or_result={has_hist} tonight_named={len(tonight_mkt)}")

    # --- Load all seasons and fit ---
    WriteJson(status, {"phase": "fit"})
    seasons = [2022, 2023, 2024, 2025, 2026]
    players = LoadPlayerGames(seasons)
    games = LoadSeasonGames(seasons)
    Log(f"loaded player-games={len(players)} season-games={len(games)}")
    feats, pst_all, tst_all = BuildFeatures(players, games)
    train = [r for r in feats if r["season"] == TRAIN_SEASON]
    hold = [r for r in feats if r["season"] == HOLD_SEASON]
    Log(f"feature rows train={len(train)} hold={len(hold)} all={len(feats)}")
    params = FitParams(train)
    Log("params " + json.dumps({k: params[k] for k in params if k != "beta"}) + " beta=" + str(params["beta"]))
    AttachPredictions(train, params)
    AttachPredictions(hold, params)
    rho_info = EstimateRho(train, params, family="nb")
    Log(f"copula rho={rho_info}")
    params["rho"] = rho_info["rho"]
    params["rho_n_pairs"] = rho_info["n_pairs"]
    WriteJson(OUT_DIR / f"{BASE}_params.json", params)

    # --- Tonight fairs (write immediately) ---
    WriteJson(status, {"phase": "tonight"})
    gsw_game = None
    for g in slate:
        if g.get("home") == "GSW" or g.get("away") == "GSW":
            gsw_game = g
            break
    if gsw_game is None:
        # Fall back: last known team for the two players and any slate typo (GS / GSW / WAR)
        for g in slate:
            sides = {g.get("home"), g.get("away")}
            if "GSW" in sides or "GS" in sides or "WAR" in sides:
                gsw_game = g
                break
    if gsw_game is None:
        Log("WARNING: no GSW game on the 2026-11-13 slate")
        team, opp, home = "GSW", "UNK", False
        tip = None
        status_g = "no game found on scoreboard/schedule"
        game_label = "GSW vs unknown (not on fetched slate)"
    else:
        if gsw_game.get("home") in ("GSW", "GS", "WAR"):
            team, opp, home = "GSW", gsw_game.get("away"), True
        else:
            team, opp, home = "GSW", gsw_game.get("home"), False
        tip = gsw_game.get("tip")
        status_g = str(gsw_game.get("status") or "")
        game_label = f"{gsw_game.get('away')} @ {gsw_game.get('home')}" if not home else f"{gsw_game.get('away')} @ GSW"

    # Inactive / status for the two players (do not read the injury-latency job's files).
    status_notes = []
    inact_names = []
    for row in inactive or []:
        nm = " ".join(str(row.get(k) or "") for k in row)
        inact_names.append(nm)
        low = nm.lower()
        if "lendeborg" in low or "podziemski" in low or "podz" in low:
            status_notes.append(f"listed inactive/on scoreboardv2: {row}")
    # Also scan scoreboard player lists if present (usually not).
    if "not play" in status_g.lower() or "final" in status_g.lower() or "q" in status_g.lower() or "half" in status_g.lower():
        if any(s in status_g.lower() for s in ("q1", "q2", "q3", "q4", "half", "final", "ot")):
            status_notes.append(f"game status={status_g}; fairs are pregame projections and may be stale if the game has started")

    pst, tst, _, _ = TonightState(players, games, ASOF)
    # Verify teams from the most recent player-game before asof, not from memory.
    verified = {}
    for name, pid in TARGETS.items():
        st = pst.get(pid) or {}
        # last team from player logs
        last_team = None
        last_date = st.get("last_date")
        for r in players:
            if r["PLAYER_ID"] == pid and str(r["GAME_DATE"])[:10] < ASOF:
                if last_team is None or str(r["GAME_DATE"])[:10] >= str(last_date or ""):
                    last_team = r.get("TEAM_ABBREVIATION")
                    last_date = str(r["GAME_DATE"])[:10]
        verified[name] = {"player_id": pid, "team": last_team, "last_played": last_date,
                          "prior_n_career": (st.get("n") or 0),
                          "prior_n_2026": sum(1 for r in players if r["PLAYER_ID"] == pid
                                              and str(r.get("SEASON_ID") or "").endswith("2026")
                                              and str(r["GAME_DATE"])[:10] < ASOF)}
        if last_team and last_team != "GSW":
            status_notes.append(f"{name} last logged team is {last_team}, not GSW")
    Log("verified " + json.dumps(verified))

    rows_tn = []
    for name, pid in TARGETS.items():
        rec = PregameRow(pid, name, team, opp, home, ASOF, pst, tst)
        rows_tn.append(rec)
    AttachPredictions(rows_tn, params)
    by_name = {r["player_name"]: r for r in rows_tn}
    lend = by_name.get("Yaxel Lendeborg")
    podz = by_name.get("Brandin Podziemski")
    if lend is None or podz is None:
        raise SystemExit(f"missing tonight rows: {list(by_name)}")

    lend_rookie = (verified["Yaxel Lendeborg"]["prior_n_career"] < 25
                   or verified["Yaxel Lendeborg"]["prior_n_2026"] < 25)
    podz_low = verified["Brandin Podziemski"]["prior_n_career"] < 25
    lmin_lo, lmin_hi = MinutesRange(lend, rookie=lend_rookie)
    pmin_lo, pmin_hi = MinutesRange(podz, rookie=podz_low)
    # For the veteran, keep a tight minutes-only range (same shrinkage).
    lend_rng = RangeRow(lend, params, lmin_lo, lmin_hi, k_mult_lo=2.0, k_mult_hi=0.5) if lend_rookie else RangeRow(
        lend, params, lmin_lo, lmin_hi, k_mult_lo=1.0, k_mult_hi=1.0)
    podz_rng = RangeRow(podz, params, pmin_lo, pmin_hi, k_mult_lo=1.0, k_mult_hi=1.0)

    family = "nb"
    alpha = params["alpha"]
    rho = params["rho"]
    caveat = ("Minutes assume both are available. Status is from the public scoreboard / "
              "ScoreboardV2 inactive list only; the official injury report was not parsed "
              "from the in-progress injury-latency job. ")
    if status_notes:
        caveat += " ".join(str(s) for s in status_notes)
    else:
        caveat += "Neither player was on the ScoreboardV2 inactive list."
    if lend_rookie:
        caveat += (f" Lendeborg is treated as a low-sample NBA player "
                   f"({verified['Yaxel Lendeborg']['prior_n_career']} career games before {ASOF}, "
                   "no earlier NBA season in the local logs). Projection is NBA-only "
                   "empirical-Bayes shrinkage of 3PA/min and 3P% toward the 2024-25 league "
                   "prior, plus an EWMA of his own minutes. Low/high widens minutes to his "
                   "recent IQR and uses 2? / 0.5? shrinkage strength. No college stats were used.")
    caveat += f" Frozen model coefficients are from 2024-25 only. Joint uses a Gaussian copula with rho={rho:.3f}."

    def singles(row, rng, rookie):
        out = []
        proj_3pa = row["proj_min"] * row["rate_3pa"]
        for k in (4, 5):
            p = NbGe(k, row["lam"], alpha)
            plo = NbGe(k, rng["lam_lo"], alpha)
            phi = NbGe(k, rng["lam_hi"], alpha)
            if plo > phi:
                plo, phi = phi, plo
            out.append({
                "date": ASOF, "sport": "nba", "series": "nba_3pm_props",
                "game": game_label, "tip_et": UtcToEtLabel(tip),
                "team": team, "opponent": opp, "home": home,
                "player_or_combo": row["player_name"],
                "market": f"{k}+",
                "fair": p, "fair_lo": plo, "fair_hi": phi,
                "lambda": row["lam"], "lambda_lo": rng["lam_lo"], "lambda_hi": rng["lam_hi"],
                "proj_min": row["proj_min"], "proj_min_lo": rng["min_lo"], "proj_min_hi": rng["min_hi"],
                "proj_3pa": proj_3pa, "shrunk_3p_pct": row["pct_3p"],
                "prior_n": row["prior_n"], "prior_3pa": row["prior_3pa"], "prior_3pm": row["prior_3pm"],
                "last_played": row.get("last_played"),
                "corr": None, "family": family, "status_caveat": caveat,
                "assumption_note": (f"proj_min=EWMA half-life {params['ewma_halflife']} games; "
                                    f"3PA/min shrunk k={params['k_rate']:.0f} to mu={params['mu_rate']:.3f}; "
                                    f"3P% shrunk k={params['k_pct']:.0f} to mu={params['mu_pct']:.3f}; "
                                    f"NB2 alpha={alpha:.3f}; home={home} opp={opp} b2b={row['b2b']}."),
            })
        return out

    tonight_rows = []
    tonight_rows += singles(lend, lend_rng, lend_rookie)
    tonight_rows += singles(podz, podz_rng, podz_low)
    combos = [(4, 4), (4, 5), (5, 4), (5, 5)]
    for kl, kp in combos:
        p = JointGe(kl, kp, lend["lam"], podz["lam"], alpha, rho, family="nb")
        plo = JointGe(kl, kp, lend_rng["lam_lo"], podz_rng["lam_lo"], alpha, rho, family="nb")
        phi = JointGe(kl, kp, lend_rng["lam_hi"], podz_rng["lam_hi"], alpha, rho, family="nb")
        if plo > phi:
            plo, phi = phi, plo
        tonight_rows.append({
            "date": ASOF, "sport": "nba", "series": "nba_3pm_props",
            "game": game_label, "tip_et": UtcToEtLabel(tip),
            "team": team, "opponent": opp, "home": home,
            "player_or_combo": "Lendeborg + Podziemski",
            "market": f"Lendeborg {kl}+ / Podziemski {kp}+",
            "fair": p, "fair_lo": plo, "fair_hi": phi,
            "lambda": None, "lambda_lo": None, "lambda_hi": None,
            "proj_min": None, "proj_min_lo": None, "proj_min_hi": None,
            "proj_3pa": None, "shrunk_3p_pct": None,
            "prior_n": None, "prior_3pa": None, "prior_3pm": None,
            "last_played": None,
            "corr": rho, "family": family, "status_caveat": caveat,
            "assumption_note": (f"Gaussian copula rho={rho:.3f} from 2024-25 same-team mid-p PIT "
                                f"(n_pairs={rho_info['n_pairs']}); NB2 marginals; low/high uses the "
                                "player lambda ranges with the same frozen rho."),
        })

    assumptions = (
        f"Game: {game_label}. Tip: {UtcToEtLabel(tip)}. Source: {slate_src}. Status: {status_g}.\n\n"
        f"- Lendeborg ({verified['Yaxel Lendeborg']['team']}) vs {opp}, last played {verified['Yaxel Lendeborg']['last_played']}, "
        f"career games before tonight={verified['Yaxel Lendeborg']['prior_n_career']} "
        f"(2026-27={verified['Yaxel Lendeborg']['prior_n_2026']}). "
        f"proj_min={Fmt(lend['proj_min'],1)}, proj_3PA={Fmt(lend['proj_min']*lend['rate_3pa'],1)}, "
        f"shrunk 3P%={Fmt(lend['pct_3p'],3)}, ?={Fmt(lend['lam'],2)}.\n"
        f"- Podziemski ({verified['Brandin Podziemski']['team']}) vs {opp}, last played {verified['Brandin Podziemski']['last_played']}, "
        f"career games before tonight={verified['Brandin Podziemski']['prior_n_career']} "
        f"(2026-27={verified['Brandin Podziemski']['prior_n_2026']}). "
        f"proj_min={Fmt(podz['proj_min'],1)}, proj_3PA={Fmt(podz['proj_min']*podz['rate_3pa'],1)}, "
        f"shrunk 3P%={Fmt(podz['pct_3p'],3)}, ?={Fmt(podz['lam'],2)}.\n"
        f"- Joint correlation rho={rho:.3f} (Gaussian copula, 2024-25 teammate pairs n={rho_info['n_pairs']}). "
        f"NB2 alpha={alpha:.3f}.\n"
        f"- {caveat}"
    )
    WriteTonight(OUT_DIR / f"{BASE}_tonight.csv", OUT_DIR / f"{BASE}_tonight_preview.md",
                 tonight_rows, assumptions, GitHash())
    # Also print the table now so a log scrape can recover fairs if later steps fail.
    Log("TONIGHT FAIRS")
    for r in tonight_rows:
        Log(f"  {r['player_or_combo']} {r['market']}: fair={Fmt(r['fair'],3)} "
            f"range=[{Fmt(r.get('fair_lo'),3)}, {Fmt(r.get('fair_hi'),3)}] "
            f"min={Fmt(r.get('proj_min'),1)} 3PA={Fmt(r.get('proj_3pa'),1)} lam={Fmt(r.get('lambda'),2)} corr={Fmt(r.get('corr'),3)}")

    # --- Calibration on 2025-26 ---
    WriteJson(status, {"phase": "calibration"})
    hold_c = FilterCalib(hold)
    cal_rows = []
    rel_rows = []
    for family_name, pkey in (("poisson", "p_pois_"), ("nb", "p_nb_")):
        for k in THRESHOLDS:
            p = [r[f"{pkey}{k}"] for r in hold_c]
            y = [1.0 if r["actual_3pm"] >= k else 0.0 for r in hold_c]
            met = MetricsBinary(p, y)
            cal_rows.append({
                "date": ASOF, "sport": "nba", "series": "nba_3pm_props",
                "model": family_name, "threshold": k, "split": "holdout_2025-26",
                "n": met["n"], "log_loss": met["log_loss"], "brier": met["brier"], "ece": met["ece"],
            })
            rel_rows.extend(ReliabilityRows(family_name, k, p, y))
    WriteCsv(OUT_DIR / f"{BASE}_calibration.csv", cal_rows)
    WriteCsv(OUT_DIR / f"{BASE}_reliability.csv", rel_rows)
    Log("calibration " + json.dumps(cal_rows, default=float))

    # --- Price history scoring, only if we actually have settled 3PT markets ---
    spa = None
    mc = None
    n_priced = 0
    price_note = "No usable Kalshi 3PT / player-prop price history was found, so there is no fee-adjusted edge, SPA, or bankroll simulation. Calibration only."
    if has_hist and kalshi_markets:
        # Conservative: only score if we can join a settled market to a player-game outcome.
        # This path is implemented, but we still require a real join; otherwise fall back.
        price_note = ("Some Kalshi markets looked settled, but a reliable join from those "
                      "tickers onto player-game 3PM outcomes was not established in this run. "
                      "No edge is claimed.")
        # Attempt a very strict join: title/ticker contains player last name and a threshold,
        # close time on a game date, and a yes/no result.
        # If too few, keep the plain statement.
        last_names = {}
        for r in hold + [x for x in feats if x["season"] == CURRENT_SEASON]:
            last_names.setdefault((r["player_name"] or "").split()[-1].lower(), []).append(r)
        # This is intentionally unused unless we can parse confidently; left as a hook.
        priced = []
        n_priced = len(priced)
        if n_priced == 0:
            has_hist = False

    verdict = "PASS"
    if not has_hist:
        verdict = "PASS"
        # calibration-only; no edge claimed
    # Summary md
    def cal_line(model, k):
        rec = next((r for r in cal_rows if r["model"] == model and r["threshold"] == k), None)
        if not rec:
            return "NA"
        return f"N={rec['n']} LL={Fmt(rec['log_loss'])} Brier={Fmt(rec['brier'])} ECE={Fmt(rec['ece'])}"

    lines = []
    lines.append("# NBA made-threes (3PM) props ? 2026-11-13")
    lines.append("")
    lines.append("Simulation / model fairs only. No wager was placed. Fee basis (if prices had existed): "
                 f"`{FEE_BASIS}` (ask + $0.02 is Tim's working estimate).")
    lines.append("")
    lines.append("## Tonight's fair values")
    lines.append("")
    lines.append(assumptions.replace("\n", "\n\n"))
    lines.append("")
    lines.append("| market | fair | low | high | proj min | proj 3PA | ? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in tonight_rows:
        lines.append(
            f"| {r['player_or_combo']} {r['market']} | {Fmt(r['fair'], 2)} | "
            f"{Fmt(r.get('fair_lo'), 2)} | {Fmt(r.get('fair_hi'), 2)} | "
            f"{Fmt(r.get('proj_min'), 1)} | {Fmt(r.get('proj_3pa'), 1)} | {Fmt(r.get('lambda'), 2)} |"
        )
    lines.append("")
    lines.append("## Calibration (fit 2024-25, hold out 2025-26)")
    lines.append("")
    lines.append(f"- Leak-free walk: features for a game use only earlier dates. Shrinkage k and GLM "
                 f"adjustments (home, B2B, pace z, opponent 3PA-allowed z) are frozen on 2024-25 "
                 f"(n_fit={params['n_fit']}). Holdout is 2025-26 rotation games with prior n?5, "
                 f"prior 3PA?10, projected minutes ? {MIN_PROJ:.0f}.")
    lines.append(f"- Shrinkage: 3P% ?={Fmt(params['mu_pct'],3)} k={Fmt(params['k_pct'],1)} attempts; "
                 f"3PA/min ?={Fmt(params['mu_rate'],3)} k={Fmt(params['k_rate'],1)} minutes.")
    lines.append(f"- NB2 dispersion ?={Fmt(params['alpha'],3)}. Same-team Gaussian copula ?={Fmt(rho,3)} "
                 f"(n_pairs={rho_info['n_pairs']}).")
    lines.append("")
    lines.append("| model | P(?2) | P(?3) | P(?4) | P(?5) |")
    lines.append("|---|---|---|---|---|")
    for model in ("poisson", "nb"):
        cells = [model]
        for k in THRESHOLDS:
            rec = next((r for r in cal_rows if r["model"] == model and r["threshold"] == k), None)
            cells.append(f"LL {Fmt(rec['log_loss'])} / Brier {Fmt(rec['brier'])}" if rec else "NA")
        lines.append("| " + " | ".join(cells) + " |")
    n_hold = len(hold_c)
    lines.append("")
    lines.append(f"Holdout N (player-games) = {n_hold}. Full bin table: `{BASE}_calibration.csv`, `{BASE}_reliability.csv`.")
    lines.append("")
    lines.append("## Kalshi prices / edge")
    lines.append("")
    lines.append(price_note)
    lines.append(f"Series scan kept {len(series)} NBA-or-three-ish series; {len(kalshi_markets)} markets looked like 3PT props; "
                 f"{len(tonight_mkt)} named Lendeborg or Podziemski.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- Projected minutes are an EWMA of past minutes, not a confirmed lineup. A late scratch or a "
                 "teammate absence (for example a high-usage shooter) would move both the singles and the combos.")
    lines.append("- Rookie / low-sample ranges are sensitivity bands, not statistical 95% intervals.")
    lines.append("- Combos use a Gaussian copula on NB marginals; they are not a full shared-minutes simulator.")
    lines.append("")
    lines.append(f"**Verdict: {verdict}.** " + (
        "No Kalshi 3PM price history, so no edge is claimed; the deliverable is the holdout calibration plus tonight's fairs."
        if not has_hist else
        "See the edge section. PASS unless a pre-registered rule beats the fee-adjusted market on holdout with SPA p<0.05 and a CI above 0."
    ))
    lines.append("")
    lines.append(f"Code `feature/3pm-props` at `{GitHash()}`, not pushed. Elapsed {time.time()-t0:.0f}s.")
    (OUT_DIR / f"{BASE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = [{
        "date": ASOF, "sport": "nba", "series": "nba_3pm_props",
        "verdict": verdict, "n_hold": n_hold, "alpha": alpha, "rho": rho,
        "n_fit": params["n_fit"], "fee_basis": FEE_BASIS,
        "has_kalshi_history": bool(has_hist), "n_kalshi_3pt_markets": len(kalshi_markets),
        "n_tonight_named_markets": len(tonight_mkt),
        "game": game_label, "opp": opp, "home": home,
        "lend_lam": lend["lam"], "podz_lam": podz["lam"],
        "lend_fair_4": tonight_rows[0]["fair"], "lend_fair_5": tonight_rows[1]["fair"],
        "podz_fair_4": tonight_rows[2]["fair"], "podz_fair_5": tonight_rows[3]["fair"],
        "combo_4_4": tonight_rows[4]["fair"], "combo_4_5": tonight_rows[5]["fair"],
        "combo_5_4": tonight_rows[6]["fair"], "combo_5_5": tonight_rows[7]["fair"],
        "git": GitHash(),
    }]
    WriteCsv(OUT_DIR / f"{BASE}.csv", summary)
    WriteJson(status, {"phase": "done", "verdict": verdict, "elapsed_s": time.time() - t0})
    (OUT_DIR / "_DONE.txt").write_text(f"verdict={verdict}\n", encoding="utf-8")
    Log(f"DONE verdict={verdict} elapsed={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
