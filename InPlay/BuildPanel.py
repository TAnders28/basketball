r"""Join ESPN game state with Kalshi and Polymarket prices on a 60-second grid per game.

    .venv\Scripts\python.exe -m InPlay.BuildPanel --seasons 2025 2026 2027 --workers 8

Writes data\inplay\panel.pkl (one row per game-minute) and data\inplay\games.pkl (one row per game).
Every price/state at grid time t is an as-of value (last observation at or before t).
"""

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from InPlay.Align import AsOf, EspnStates, EtDate, IsoTs, KalshiMid, SeasonOf, Team, TipAndEnd

ROOT = os.path.join("data", "inplay")
PRE_MIN = 30
POST_MIN = 3


def LoadIndexes():
    k = json.load(open(os.path.join(ROOT, "kalshi", "markets.json"), encoding="utf-8"))
    kidx = {}
    for r in k:
        if r["is_home"]:
            kidx.setdefault((r["date_et"], Team(r["home"]), Team(r["away"])), r)
    pidx = {}
    pm_path = os.path.join(ROOT, "polymarket", "markets.json")
    if os.path.exists(pm_path):
        for r in json.load(open(pm_path, encoding="utf-8")):
            pidx.setdefault((Team(r["home"]), Team(r["away"])), []).append(r)
    return kidx, pidx


def KalshiSeries(ticker):
    path = os.path.join(ROOT, "kalshi", "candles", ticker + ".json")
    if not os.path.exists(path):
        return None
    candles = json.load(open(path, encoding="utf-8"))["candles"]
    candles.sort(key=lambda c: c["t"])
    return candles


def PmSeries(slug):
    path = os.path.join(ROOT, "polymarket", "prices", slug + ".json")
    if not os.path.exists(path):
        return None
    hist = sorted(json.load(open(path, encoding="utf-8"))["history"], key=lambda h: h["t"])
    return hist or None


def BuildGame(game, kmeta, pmeta):
    """Return (game_row, list of grid rows) or (game_row with reason, [])."""
    states = EspnStates(game)
    tip, end = TipAndEnd(states)
    sched = IsoTs(game.get("date"))
    home, away = Team(game.get("home")), Team(game.get("away"))
    hw = game.get("home_won")
    g = {"game_id": game.get("id"), "season": SeasonOf(sched) if sched else None, "season_type": game.get("season_type"),
         "date_et": EtDate(sched) if sched else None, "home": home, "away": away, "sched_ts": sched,
         "tip_ts": tip, "end_ts": end, "home_won": None if hw is None else int(bool(hw)),
         "kalshi_ticker": kmeta["ticker"] if kmeta else None, "pm_slug": pmeta["slug"] if pmeta else None,
         "n_plays": len(states), "reason": None}
    pick = (game.get("pickcenter") or [{}])[0] if game.get("pickcenter") else {}
    g["book_home_ml"], g["book_away_ml"], g["book_spread"] = pick.get("homeML"), pick.get("awayML"), pick.get("spread")
    g["book_total"] = pick.get("overUnder")
    if tip is None or end is None or end - tip < 3600 or hw is None:
        g["reason"] = "bad_espn"
        return g, []
    kc = KalshiSeries(kmeta["ticker"]) if kmeta else None
    pm = PmSeries(pmeta["slug"]) if pmeta else None
    if kmeta and kmeta.get("result") in ("yes", "no"):
        g["kalshi_result_home"] = 1 if kmeta["result"] == "yes" else 0
    if not kc and not pm:
        g["reason"] = "no_prices"
        return g, []
    st_t = [s["t"] for s in states]
    k_t = [c["t"] for c in kc] if kc else []
    p_t = [h["t"] for h in pm] if pm else []
    start = (tip // 60) * 60 - PRE_MIN * 60
    stop = end + POST_MIN * 60
    rows = []
    t = start
    while t <= stop:
        s = AsOf(st_t, states, t)
        c = AsOf(k_t, kc, t) if kc else None
        h = AsOf(p_t, pm, t) if pm else None
        r = {"game_id": g["game_id"], "t": t, "inplay": int(tip <= t <= end),
             "el": s["el"] if s and t >= tip else 0.0, "per": s["per"] if s and t >= tip else 0,
             "sd": (s["hs"] - s["as"]) if s and t >= tip else 0, "wp": s["wp"] if s and t >= tip else None,
             "run": s.get("run", 0) if s and t >= tip else 0,
             "to_age": t - s["last_to"] if s and s.get("last_to") is not None else None,
             "k_bid": None, "k_ask": None, "k_mid": None, "k_last": None, "k_vol": 0.0, "k_age": None, "k_trlo": None, "k_trhi": None,
             "pm": None, "pm_age": None}
        if c is not None:
            r["k_bid"], r["k_ask"], r["k_last"] = c["bid"], c["ask"], c["last"]
            r["k_mid"] = KalshiMid(c["bid"], c["ask"])
            r["k_vol"] = c["vol"] if c["t"] == t else 0.0
            r["k_age"] = t - c["t"]
            if c["t"] == t:
                r["k_trlo"], r["k_trhi"] = c.get("tr_lo"), c.get("tr_hi")
        if h is not None:
            r["pm"], r["pm_age"] = h["p"], t - h["t"]
        rows.append(r)
        t += 60
    # pregame price: as-of one minute before tip
    c0 = AsOf(k_t, kc, tip - 60) if kc else None
    h0 = AsOf(p_t, pm, tip - 60) if pm else None
    g["k_pre"] = KalshiMid(c0["bid"], c0["ask"]) if c0 else None
    g["pm_pre"] = h0["p"] if h0 else None
    g["n_kbars"] = len(kc) if kc else 0
    g["n_pm"] = len(pm) if pm else 0
    return g, rows


def _Work(args):
    path, kidx, pidx = args
    game = json.load(open(path, encoding="utf-8"))
    sched = IsoTs(game.get("date"))
    if sched is None:
        return {"game_id": game.get("id"), "reason": "no_date"}, []
    home, away = Team(game.get("home")), Team(game.get("away"))
    kmeta = kidx.get((EtDate(sched), home, away))
    pmeta = None
    for cand in pidx.get((home, away), []):
        if abs(cand["start_ts"] - sched) < 6 * 3600:
            pmeta = cand
            break
    return BuildGame(game, kmeta, pmeta)


def Main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026, 2027])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "panel.pkl"))
    args = ap.parse_args(argv)
    kidx, pidx = LoadIndexes()
    paths = []
    for s in args.seasons:
        paths += sorted(glob.glob(os.path.join(ROOT, "espn", "games", str(s), "*.json")))
    games, rows = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, 10)) as ex:
        for g, r in ex.map(_Work, [(p, kidx, pidx) for p in paths], chunksize=20):
            games.append(g)
            rows += r
    gdf = pd.DataFrame(games)
    pdf = pd.DataFrame(rows)
    gdf.to_pickle(args.out.replace("panel", "games"))
    pdf.to_pickle(args.out)
    print(gdf.groupby(["season"]).agg(n=("game_id", "size"), kalshi=("kalshi_ticker", "count"), pm=("pm_slug", "count")))
    print(gdf["reason"].value_counts(dropna=False))
    print(f"panel rows {len(pdf)}")


if __name__ == "__main__":
    sys.exit(Main())