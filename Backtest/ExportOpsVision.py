"""Export NBA calibration, $10k bankroll and edge CSVs in Degen Bot's common format.

One script, one run. Walk-forward stack_global and Elo v1 come from the frozen
worktree (feature/predictday-stack-default, 6df88fe). Prices are Kalshi T-60
(home YES candles). Staking is flat100 and qkelly from $10,000. Fee is Tim's
$0.02 estimate. Bankroll x-axis uses Backtest.BankrollPlot (season rule).

.venv\\Scripts\\python.exe -m Backtest.ExportOpsVision
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

MAIN = Path(r"E:\AgentWorkspaces\Basketball\basketball")
WORKTREE = Path(r"E:\AgentWorkspaces\Basketball\basketball-wt-v3")
KALSHI_MARKETS = MAIN / "data" / "inplay" / "kalshi" / "markets.json"
KALSHI_CANDLES = MAIN / "data" / "inplay" / "kalshi" / "candles"
GAMES_META = MAIN / "data" / "inplay" / "games.pkl"
CAL_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_calibration")
BANK_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_bankroll-10k")
CAL_BASE = "2026-11-13_calibration"
BANK_BASE = "2026-11-13_bankroll-10k"
FEE = 0.02
FEE_BASIS = "tim_est_0.02"
PRICE_BASIS = "kalshi_T60"
EDGE_CUT = 0.03
START_BANK = 10000.0
FLAT_STAKE = 100.0
QKELLY_FRAC = 0.25
QKELLY_CAP = 0.02
STALE_SEC = 1800
T60_SEC = 3600
ET = ZoneInfo("America/New_York")
BINS = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]
RECORD_FROM = "2024-11-15"
HOLDOUT_SEASON = "2025-26"
FORWARD_SEASON = "2026-27"
# File year 2025 = 2025-26 holdout; file year 2026 = 2026-27 forward.
SEASON_FILES = {2025: HOLDOUT_SEASON, 2026: FORWARD_SEASON}
SPLIT_OF = {HOLDOUT_SEASON: "holdout", FORWARD_SEASON: "forward"}
# Official PredictDay window is three seasons ending at the game's file year.
WALK_WINDOWS = {
    HOLDOUT_SEASON: [2023, 2024, 2025],
    FORWARD_SEASON: [2024, 2025, 2026],
}


def PreparePath() -> None:
    """Import stack_global from the frozen worktree; BankrollPlot from this checkout.

    Run this file as a script (not python -m Backtest.ExportOpsVision) so the
    Backtest package is not already bound to the current checkout.
    """
    os.chdir(MAIN)
    # Drop any Backtest/Matchups already imported from the wrong tree.
    for name in list(sys.modules):
        if name == 'Backtest' or name.startswith('Backtest.') or name == 'Matchups' or name.startswith('Matchups.'):
            del sys.modules[name]
    # Worktree first so Matchups / Backtest.StackDay / Model are 6df88fe.
    while str(WORKTREE) in sys.path:
        sys.path.remove(str(WORKTREE))
    sys.path.insert(0, str(WORKTREE))


def LoadBankrollPlot():
    path = MAIN / "Backtest" / "BankrollPlot.py"
    spec = importlib.util.spec_from_file_location("BasketbotBankrollPlot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def IsoUtc(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat()


def IsoEt(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), ET).isoformat()


def ClipP(p: float) -> float:
    return min(max(float(p), 1e-15), 1.0 - 1e-15)


def Mid(bid, ask):
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


def LoadJson(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def LoadSeasonGames() -> list:
    """Completed OOS games from the official season files (regular, play-in, playoffs)."""
    rows = []
    for year, label in SEASON_FILES.items():
        games = LoadJson(MAIN / "data" / "season" / f"games_{year}.json")
        for g in games:
            if g.get("home_score") is None or g.get("away_score") is None:
                continue
            if int(g["home_score"]) == int(g["away_score"]):
                g = dict(g)
                g["_tie"] = True
            else:
                g = dict(g)
                g["_tie"] = False
            g["season_label"] = label
            g["split"] = SPLIT_OF[label]
            g["file_year"] = year
            rows.append(g)
    return sorted(rows, key=lambda g: (g["game_date"], g["game_id"]))


def LoadSchedIndex():
    import pickle
    import pandas as pd
    frame = pickle.loads(GAMES_META.read_bytes())
    out = {}
    for rec in frame.itertuples(index=False):
        key = (str(rec.date_et)[:10], rec.home, rec.away)
        sched = None if pd.isna(rec.sched_ts) else int(rec.sched_ts)
        ticker = None if pd.isna(rec.kalshi_ticker) else str(rec.kalshi_ticker)
        out[key] = {"sched_ts": sched, "kalshi_ticker": ticker}
    return out


def LoadMarketIndex():
    markets = LoadJson(KALSHI_MARKETS)
    by_event = defaultdict(dict)
    for m in markets:
        key = (m["date_et"], m["home"], m["away"])
        side = "home" if m.get("is_home") else "away"
        by_event[key][side] = m
        by_event[key]["event_ticker"] = m.get("event_ticker")
    return by_event


def QuoteAtT60(ticker: str, sched_ts: int):
    path = KALSHI_CANDLES / f"{ticker}.json"
    if not ticker:
        return None, "no_ticker"
    if not path.exists():
        return None, "no_candle_file"
    candles = LoadJson(path).get("candles") or []
    if not candles:
        return None, "empty_candles"
    target = int(sched_ts) - T60_SEC
    before = [c for c in candles
              if c.get("t") is not None and int(c["t"]) <= target
              and c.get("bid") is not None and c.get("ask") is not None]
    if not before:
        return None, "no_quote_at_or_before_T-60"
    c = before[-1]
    age = target - int(c["t"])
    if age > STALE_SEC:
        return None, "stale_quote_>30min"
    bid, ask = float(c["bid"]), float(c["ask"])
    if not (0.0 <= bid < 1.0) or not (0.0 < ask <= 1.0) or ask <= bid < 0:
        # allow bid==ask; reject inverted or empty
        if ask < bid or ask <= 0 or bid < 0 or ask > 1 or bid > 1:
            return None, "bad_bid_ask"
    mid = Mid(bid, ask)
    if mid is None or not (0.0 < mid < 1.0):
        return None, "bad_mid"
    return {
        "bid": bid, "ask": ask, "mid": mid,
        "t": int(c["t"]), "age_s": age,
    }, None


def AttachPrices(games: list, sched_index: dict, market_index: dict):
    """Join schedule + Kalshi T-60 home-YES quotes. Away ask is implied as 1-home_bid."""
    priced, dropped, ties = [], [], 0
    for g in games:
        if g.get("_tie"):
            ties += 1
            dropped.append({"game_id": g["game_id"], "reason": "tie"})
            continue
        key = (g["game_date"], g["home_team"], g["away_team"])
        sched = sched_index.get(key, {})
        markets = market_index.get(key, {})
        home_m = markets.get("home")
        reason = None
        sched_ts = sched.get("sched_ts")
        ticker = sched.get("kalshi_ticker") or (home_m["ticker"] if home_m else None)
        if home_m and home_m.get("result") not in ("yes", "no"):
            reason = f"kalshi_result_{home_m.get('result')}"
        elif sched_ts is None:
            occ = home_m.get("occurrence_datetime") if home_m else None
            if occ:
                sched_ts = int(datetime.fromisoformat(occ.replace("Z", "+00:00")).timestamp())
            else:
                reason = "no_sched_ts"
        if reason is None and not ticker:
            reason = "no_ticker"
        quote, qreason = (None, None)
        if reason is None:
            quote, qreason = QuoteAtT60(ticker, sched_ts)
            if quote is None:
                reason = qreason
        if reason:
            dropped.append({"game_id": g["game_id"], "date": g["game_date"],
                            "event": f"{g['away_team']} @ {g['home_team']}", "reason": reason})
            g["priced"] = False
            g["drop_reason"] = reason
            g["sched_ts"] = sched_ts
            continue
        home_win = int(g["home_score"] > g["away_score"])
        g.update({
            "priced": True,
            "sched_ts": int(sched_ts),
            "start_time": IsoEt(sched_ts),
            "price_time": IsoUtc(quote["t"]),
            "ticker": ticker,
            "home_bid": quote["bid"],
            "home_ask": quote["ask"],
            "home_mid": quote["mid"],
            "away_ask": 1.0 - quote["bid"],
            "market_prob_devig_home": quote["mid"],
            "home_win": home_win,
        })
        priced.append(g)
    return priced, dropped, ties


def StackWindow(seasons: list, progress=None) -> list:
    """Leak-free stack_global + Elo v1 rows for games in the three-season window."""
    from Backtest.StackDay import (EloHistory, FillMember, MemberNames, WalkMembers)
    from Matchups.V2.Base import REGISTRY, LoadGames
    from Matchups.V3 import Post as PP

    games = [g for g in LoadGames(seasons)]
    # Also include non-002 completed games (play-in / playoffs) in this window
    # so postseason is scored leak-free. LoadGames is regular-season only.
    from Matchups import Data as D
    have = {g["game_id"] for g in games}
    extra = []
    for s in seasons:
        for g in D.AllGames(s):
            if g["game_id"] in have:
                continue
            if g.get("home_score") is None:
                continue
            g = dict(g)
            g["season"] = s
            extra.append(g)
    if extra:
        games = sorted(games + extra, key=lambda g: (g["game_date"], g["game_id"]))
        if progress:
            progress(f"window {seasons}: {len(extra)} postseason/other games added, total {len(games)}")
    approaches = {name: REGISTRY[name]() for name in MemberNames()}
    if progress:
        progress(f"window {seasons}: walking {len(approaches)} members over {len(games)} games")
    walked = WalkMembers(approaches, games, record_from=RECORD_FROM, progress=progress)
    if progress:
        progress(f"window {seasons}: WalkMembers done")
    until = (datetime.fromisoformat(games[-1]["game_date"]).date().toordinal())
    until_s = games[-1]["game_date"]
    # EloHistory `until` is exclusive; bump one day past the last game.
    last = datetime.fromisoformat(until_s).date()
    from datetime import timedelta
    until_excl = (last + timedelta(days=1)).isoformat()
    elo = EloHistory(seasons, RECORD_FROM, until_excl)
    if progress:
        progress(f"window {seasons}: EloHistory done, {len(elo['rows'])} rows")
    member_rows = dict(walked["rows"])
    member_rows["elo_phase1"] = elo["rows"]
    stacked, names, _w, _wc = PP.ContextStack(member_rows, contexts={}, contextual=False)
    if progress:
        progress(f"window {seasons}: ContextStack done, {len(stacked)} rows")
    by_id = {}
    elo_by = {r["game_id"]: r for r in elo["rows"]}
    for r in stacked:
        eid = r["game_id"]
        er = elo_by.get(eid, {})
        by_id[eid] = {
            "game_id": eid,
            "game_date": r["game_date"],
            "stack_global_home_prob": float(r["p_home"]),
            "elo_predictday_v1_home_prob": float(er.get("p_home", r.get("p_home", 0.5))),
            "home": r.get("home"),
            "away": r.get("away"),
            "home_win_walk": r.get("home_win"),
        }
    if progress:
        progress(f"window {seasons}: stacked {len(by_id)} games, members={names}")
    return list(by_id.values())


def MergeForecasts(oos_games: list, stacked_rows: list) -> list:
    by_id = {r["game_id"]: r for r in stacked_rows}
    out, missing = [], []
    for g in oos_games:
        r = by_id.get(g["game_id"])
        if not r:
            g = dict(g)
            g["priced"] = False
            g["drop_reason"] = "no_model_forecast"
            missing.append(g)
            continue
        g = dict(g)
        g["stack_global_home_prob"] = r["stack_global_home_prob"]
        g["elo_predictday_v1_home_prob"] = r["elo_predictday_v1_home_prob"]
        out.append(g)
    return out, missing


def ChosenSide(model_home: float, mkt_home: float):
    """Side with the larger fee-adjusted edge vs de-vig mid."""
    home_edge = model_home - mkt_home - FEE
    away_edge = (1.0 - model_home) - (1.0 - mkt_home) - FEE
    if home_edge >= away_edge:
        return "home", home_edge, model_home, mkt_home
    return "away", away_edge, 1.0 - model_home, 1.0 - mkt_home


def FillFor(side: str, game: dict) -> float:
    return game["home_ask"] if side == "home" else game["away_ask"]


def PnlFlat(fill: float, result: int, stake: float = FLAT_STAKE) -> float:
    cost = fill + FEE
    if cost <= 0:
        return 0.0
    contracts = stake / cost
    return contracts * result - stake


def EdgeRows(priced: list) -> list:
    rows = []
    for g in priced:
        p = g["stack_global_home_prob"]
        m = g["market_prob_devig_home"]
        side, edge, model_side, mkt_side = ChosenSide(p, m)
        fill = FillFor(side, g)
        result = g["home_win"] if side == "home" else 1 - g["home_win"]
        edge_vs_fill = model_side - fill - FEE
        rows.append({
            "date": g["game_date"],
            "start_time": g["start_time"],
            "sport": "nba",
            "season": g["season_label"],
            "split": g["split"],
            "event_id": g["game_id"],
            "event": f"{g['away_team']} @ {g['home_team']}",
            "side": side,
            "model": "stack_global",
            "model_prob": model_side,
            "market_prob_devig": mkt_side,
            "fill_price": fill,
            "price_basis": PRICE_BASIS,
            "price_time": g["price_time"],
            "fee": FEE,
            "fee_basis": FEE_BASIS,
            "edge": edge,
            "edge_vs_fill": edge_vs_fill,
            "qualifies": bool(edge >= EDGE_CUT),
            "result": int(result),
            "pnl_at_flat_stake": PnlFlat(fill, int(result)),
            "_start_ts": g["sched_ts"],
            "_model_home": p,
        })
    return rows


def WalkBankroll(edge_rows: list, all_days: list, strategy: str):
    """One continuous bankroll across OOS seasons. Returns per-day rows + path stats."""
    by_day = defaultdict(list)
    for r in edge_rows:
        by_day[r["date"]].append(r)
    for d in by_day:
        by_day[d].sort(key=lambda r: r["_start_ts"])
    equity = START_BANK
    peak = START_BANK
    bets_cum = 0
    ruined = False
    day_rows = []
    trade_pnls = []  # (stake, pnl) for qualifying bets, for ROI CI
    for day in all_days:
        start_of_day = equity
        bets_day = 0
        day_events = by_day.get(day, [])
        for r in day_events:
            if ruined or equity < FLAT_STAKE:
                ruined = True
                continue
            if strategy == "market_fav":
                # Favourite = higher de-vig mid; always bet (priced games only).
                # Rebuild fav from stored home mid: market_prob_devig is the chosen side.
                # Use the market favourite, not the model side.
                # We stored model-chosen fields; recover home mid from side.
                if r["side"] == "home":
                    home_m = r["market_prob_devig"]
                else:
                    home_m = 1.0 - r["market_prob_devig"]
                if home_m >= 0.5:
                    side, fill_side = "home", "home"
                    # fill of home is not in the row if model chose away
                    fill = r["fill_price"] if r["side"] == "home" else (1.0 - (1.0 - r["fill_price"]) if False else None)
                # Safer: skip this approach here and handle market_fav separately.
        # handled below
        day_rows.append(day)
    return day_rows


def MarketFavFields(g: dict):
    m = g["market_prob_devig_home"]
    if m >= 0.5:
        side, fill, result = "home", g["home_ask"], g["home_win"]
        model_unused = m
    else:
        side, fill, result = "away", g["away_ask"], 1 - g["home_win"]
        model_unused = 1.0 - m
    return side, fill, int(result)


def BankrollPaths(priced: list, all_days: list, edge_by_id: dict):
    priced_by_day = defaultdict(list)
    for g in priced:
        priced_by_day[g["game_date"]].append(g)
    for d in priced_by_day:
        priced_by_day[d].sort(key=lambda g: g["sched_ts"])

    def Run(strategy: str):
        equity = START_BANK
        peak = START_BANK
        bets_cum = 0
        ruined = False
        ruin_day = None
        rows = []
        stakes_pnls = []
        for day in all_days:
            start_of_day = equity
            bets_day = 0
            for g in priced_by_day.get(day, []):
                if ruined or equity < FLAT_STAKE:
                    if not ruined:
                        ruined = True
                        ruin_day = day
                    continue
                if strategy == "market_fav":
                    side, fill, result = MarketFavFields(g)
                    stake = FLAT_STAKE
                    if equity < stake:
                        ruined = True
                        ruin_day = day
                        continue
                    pnl = PnlFlat(fill, result, stake)
                    equity = equity + pnl
                    bets_day += 1
                    bets_cum += 1
                    stakes_pnls.append((stake, pnl))
                    continue
                er = edge_by_id[g["game_id"]]
                if not er["qualifies"]:
                    continue
                fill = er["fill_price"]
                result = er["result"]
                p = er["model_prob"]
                c = fill + FEE
                if c >= 1.0:
                    # Negative expected even at p=1 if fee pushes cost over $1; still mark the bet.
                    pass
                if strategy == "flat100":
                    stake = FLAT_STAKE
                    if equity < stake:
                        ruined = True
                        ruin_day = day
                        continue
                    pnl = PnlFlat(fill, result, stake)
                else:
                    if c >= 1.0 or c <= 0:
                        kelly = 0.0
                    else:
                        kelly = (p - c) / (1.0 - c)
                    stake = max(0.0, QKELLY_FRAC * kelly * start_of_day)
                    stake = min(stake, QKELLY_CAP * start_of_day, equity)
                    stake = math.floor(stake * 100.0 + 1e-9) / 100.0
                    if stake < 1.0:
                        continue
                    contracts = stake / c
                    pnl = contracts * result - stake
                equity = equity + pnl
                bets_day += 1
                bets_cum += 1
                stakes_pnls.append((stake, pnl))
            if equity > peak:
                peak = equity
            dd = 0.0 if peak == 0 else min(0.0, equity / peak - 1.0)
            # season/split of this calendar day
            season = next((g["season_label"] for g in priced_by_day.get(day, [])), None)
            if season is None:
                # unpriced-only day: infer from all_days map later
                season = _DAY_SEASON[day]
            rows.append({
                "date": day,
                "sport": "nba",
                "series": f"nba_{strategy}_{PRICE_BASIS}",
                "strategy": strategy,
                "price_basis": PRICE_BASIS,
                "season": season,
                "split": SPLIT_OF[season],
                "bets_day": bets_day,
                "bets_cum": bets_cum,
                "equity": equity,
                "drawdown_pct": dd,
            })
        return {
            "rows": rows, "end": equity, "stakes_pnls": stakes_pnls,
            "ruined": ruined, "ruin_day": ruin_day, "max_dd": min((r["drawdown_pct"] for r in rows), default=0.0),
            "n_bets": bets_cum,
        }

    return {name: Run(name) for name in ("flat100", "qkelly", "market_fav")}


_DAY_SEASON = {}


def BootstrapRoi(stakes_pnls, reps=10000, seed=7, level=0.90):
    if not stakes_pnls:
        return {"roi": None, "lo": None, "hi": None, "n": 0}
    pnls = [p for _s, p in stakes_pnls]
    stakes = [s for s, _p in stakes_pnls]
    total_stake = sum(stakes)
    point = sum(pnls) / total_stake if total_stake else None
    rng = random.Random(seed)
    n = len(stakes_pnls)
    dist = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        s = sum(stakes[i] for i in idx)
        if s <= 0:
            continue
        dist.append(sum(pnls[i] for i in idx) / s)
    dist.sort()
    alpha = (1.0 - level) / 2.0
    lo = dist[int(alpha * len(dist))] if dist else None
    hi = dist[min(len(dist) - 1, int((1.0 - alpha) * len(dist)))] if dist else None
    return {"roi": point, "lo": lo, "hi": hi, "n": n}


def Metrics(pairs):
    """pairs = [(p_home, y_home), ...]"""
    if not pairs:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None, "ece": None}
    n = len(pairs)
    acc = sum(int((p >= 0.5) == y) for p, y in pairs) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = 0.0
    for p, y in pairs:
        p = ClipP(p)
        ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    ll /= n
    # ECE, 10 fixed bins
    ece = 0.0
    for lo, hi in BINS:
        if lo == 0.9:
            cell = [(p, y) for p, y in pairs if lo <= p <= hi]
        else:
            cell = [(p, y) for p, y in pairs if lo <= p < hi]
        if not cell:
            continue
        mp = sum(p for p, _y in cell) / len(cell)
        obs = sum(y for _p, y in cell) / len(cell)
        ece += (len(cell) / n) * abs(mp - obs)
    return {"n": n, "accuracy": acc, "brier": brier, "log_loss": ll, "ece": ece}


def CalibrationRows(priced: list, asof: dict):
    """asof[season_or_pooled] = last game date."""
    models = [
        ("stack_global", "stack_global_home_prob"),
        ("elo_predictday_v1", "elo_predictday_v1_home_prob"),
        ("market_kalshi_T60", "market_prob_devig_home"),
    ]
    groups = defaultdict(list)
    for g in priced:
        for model, field in models:
            groups[(g["season_label"], g["split"], model, field)].append(g)
        for model, field in models:
            groups[("pooled_oos", "holdout" if False else "pooled", model, field)].append(g)
    # fix pooled split label
    cal, summary = [], []
    seen_pooled = set()
    # rebuild groups cleanly
    groups = defaultdict(list)
    for g in priced:
        for model, field in models:
            groups[(g["season_label"], g["split"], model, field)].append(g)
            groups[("pooled_oos", "forward" if False else "holdout", model, field)]
    groups = defaultdict(list)
    for g in priced:
        for model, field in models:
            groups[(g["season_label"], g["split"], model, field)].append(g)
            groups[("pooled_oos", "pooled_oos", model, field)].append(g)

    def emit(season, split, model, field, games):
        last = asof.get(season) or (max(x["game_date"] for x in games) if games else "")
        pairs = [(float(g[field]), int(g["home_win"])) for g in games]
        # bins
        for lo, hi in BINS:
            if lo == 0.9:
                cell = [(p, y) for p, y in pairs if lo <= p <= hi]
            else:
                cell = [(p, y) for p, y in pairs if lo <= p < hi]
            n = len(cell)
            if n == 0:
                mean_pred, obs = "", ""
            else:
                mean_pred = sum(p for p, _y in cell) / n
                obs = sum(y for _p, y in cell) / n
            cal.append({
                "date": last, "sport": "nba", "series": f"nba_{model}", "model": model,
                "season": season, "split": split if season != "pooled_oos" else "pooled_oos",
                "bin_lo": lo, "bin_hi": hi, "n": n,
                "mean_pred": mean_pred, "obs_rate": obs,
            })
        met = Metrics(pairs)
        summary.append({
            "date": last, "sport": "nba", "series": f"nba_{model}", "model": model,
            "season": season, "split": split if season != "pooled_oos" else "pooled_oos",
            "n": met["n"], "accuracy": met["accuracy"], "brier": met["brier"],
            "log_loss": met["log_loss"], "ece": met["ece"],
        })

    order = [(HOLDOUT_SEASON, "holdout"), (FORWARD_SEASON, "forward"), ("pooled_oos", "pooled_oos")]
    for season, split in order:
        for model, field in models:
            key = (season, split if season != "pooled_oos" else "pooled_oos", model, field)
            emit(season, split, model, field, groups.get(key, []))
    return cal, summary


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
                    # Footbot/Basebot write Python-style True/False, not true/false.
                    out[k] = "True" if v else "False"
                elif isinstance(v, float):
                    # Match their 0.0 / 10000.0 (plain :g would write 0 / 10000).
                    if math.isfinite(v) and v == math.trunc(v):
                        out[k] = f"{v:.1f}"
                    else:
                        out[k] = f"{v:.10g}"
                else:
                    out[k] = v
            w.writerow(out)


def Fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def WriteCalibrationMd(path: Path, summary: list, priced: list, dropped: list, ties: int,
                       walk_info: str, hash_main: str, hash_wt: str) -> None:
    by = {(r["model"], r["season"]): r for r in summary}
    lines = []
    a = lines.append
    a("# NBA calibration vs Kalshi T-60 (2026-11-13)")
    a("")
    a("Simulation only. No wagers were placed.")
    a("")
    a("## Models")
    a(f"- Headline: `stack_global`, leak-free walk-forward from worktree `E:\\AgentWorkspaces\\Basketball\\basketball-wt-v3` at `6df88fe` (`feature/predictday-stack-default`). Freeze date: the commit date of 6df88fe (worktree HEAD `{hash_wt}`).")
    a("- Shadow: `elo_predictday_v1` (phase-1 Elo replica mixed as a stack member and reported on its own).")
    a("- Market: `market_kalshi_T60`, the Kalshi home-YES mid at T-60, on the **same games**.")
    a("- The walk uses the official three-season windows and weekly EM mix (`contextual=False`, no EMOS). Postseason and play-in games are included leak-free so the 2025-26 holdout covers the full season; official PredictDay `LoadGames` is regular-season 002 only.")
    a(f"- Code: worktree `{hash_wt}`; BankrollPlot / this exporter from `{hash_main}`.")
    a("")
    a("## Splits")
    a("- `holdout` = 2025-26 (season file `games_2025.json`, 2025-10-21 to 2026-06-13).")
    a("- `forward` = 2026-27 so far (season file `games_2026.json`, 2026-10-20 to 2026-11-11).")
    a("- `pooled_oos` = holdout + forward. Tune / 2024-25 playoffs are not included.")
    n_h = sum(1 for g in priced if g["split"] == "holdout")
    n_f = sum(1 for g in priced if g["split"] == "forward")
    a(f"- Priced N used for every model: holdout {n_h}, forward {n_f}, pooled {n_h + n_f}.")
    a("")
    a("## Headline metrics (home-team probability)")
    a("")
    a("| model | season | split | N | log_loss | Brier | accuracy | ECE |")
    a("|---|---|---|---:|---:|---:|---:|---:|")
    for r in summary:
        a(f"| {r['model']} | {r['season']} | {r['split']} | {r['n']} | {Fmt(r['log_loss'], 4)} | {Fmt(r['brier'], 4)} | {Fmt(r['accuracy'], 3)} | {Fmt(r['ece'], 4)} |")
    a("")
    a("## Dropped prices")
    a(f"- Ties / voids from the NBA log: {ties}.")
    a(f"- Games dropped from calibration, bankroll and edges: {len(dropped)} (see reasons below).")
    reasons = Counter(d["reason"] for d in dropped)
    for reason, n in reasons.most_common():
        a(f"- `{reason}`: {n}")
    a("")
    a("## Caveats")
    a("- **Injury:** `stack_global` and Elo v1 are not injury-adjusted. Official injury reports can move the market after these forecasts would have been locked.")
    a("- Only the Kalshi **home YES** contract was pulled. `market_prob_devig` is the home mid (away = 1-mid). The away-side taker fill is implied as `1 - home_bid` (no separate away book).")
    a("- Candles have **no size**. Fills at the T-60 ask are optimistic for large stakes.")
    a("- The $0.02 fee is **Tim's working estimate**, not a verified exchange fee (`fee_basis=tim_est_0.02`).")
    a("- T-60 is the last 1-minute candle at or before scheduled start minus 60 minutes, dropped if that quote is older than 30 minutes. Scheduled start is `sched_ts` from the in-play game table (NBA schedule).")
    a("- Leak test: member forecasts are produced before that day's results are applied; mixture weights are fit only on earlier days (same path as `Matchups.V3.Post.ContextStack` with `contextual=False`).")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def WriteBankrollMd(path: Path, paths: dict, roi: dict, priced: list, dropped: list,
                    ties: int, all_days: list, hash_main: str, hash_wt: str) -> None:
    lines = []
    a = lines.append
    a("# NBA $10k bankroll, Kalshi T-60 (2026-11-13)")
    a("")
    a("Simulation only. No wagers were placed. Default call is PASS unless a validated edge exists.")
    a("")
    a("## Setup")
    a("- Start $10,000, one continuous bankroll across out-of-sample seasons, no reset.")
    a("- `flat100`: flat $100 (1% of start) on the side with the larger edge, only if edge >= 0.03 after Tim's $0.02 fee. Contracts = 100 / (ask + 0.02). No compounding. Stop if equity < $100.")
    a("- `qkelly`: 0.25 x Kelly at cost c = ask + 0.02, times start-of-day equity, capped at 2% of start-of-day equity, same qualify rule, compounding, no reset.")
    a("- `market_fav`: flat $100 on the Kalshi favourite every priced game (fee/spread reference).")
    a("- Headline model: `stack_global` at `6df88fe`. Price basis: `kalshi_T60`. Fee basis: `tim_est_0.02` (Tim's estimate, not a verified fee).")
    a("- Edge = model_prob - de-vig mid - 0.02. P&L uses the side ask (away ask implied as 1-home_bid).")
    a(f"- Code: worktree `{hash_wt}`; BankrollPlot / exporter `{hash_main}`.")
    a("")
    a("## Splits")
    a("- `holdout` = 2025-26 (2025-10-21 to 2026-06-13).")
    a("- `forward` = 2026-27 so far (2026-10-20 to 2026-11-11).")
    a(f"- Game days in the bankroll file (bets or not): {len(all_days)}, first {all_days[0] if all_days else 'n/a'}, last {all_days[-1] if all_days else 'n/a'}. Multi-season `x` is a game-day index that skips the 2026 summer; $10,000 is implicit at x=0 (first game day is x=1).")
    n_h = sum(1 for g in priced if g["split"] == "holdout")
    n_f = sum(1 for g in priced if g["split"] == "forward")
    a(f"- Priced events: holdout {n_h}, forward {n_f}, pooled {n_h + n_f}.")
    a("")
    a("## Results")
    a("")
    a("| strategy | N bets | ending bankroll | ROI per $ staked [90% event bootstrap CI] | max drawdown | ruined |")
    a("|---|---:|---:|---|---:|---|")
    for name in ("flat100", "qkelly", "market_fav"):
        p = paths[name]
        r = roi[name]
        roi_s = "n/a" if r["roi"] is None else f"{r['roi']:+.3f} [{Fmt(r['lo'], 3)}, {Fmt(r['hi'], 3)}]"
        a(f"| {name} | {p['n_bets']} | ${p['end']:,.2f} | {roi_s} | {p['max_dd']:.3f} | {'yes at ' + p['ruin_day'] if p['ruined'] else 'no'} |")
    a("")
    # Qualify counts
    n_qual = sum(1 for g in priced if edge_qual(g))
    a(f"- Qualifying opportunities (edge >= 0.03 on `stack_global`, Tim's $0.02 fee): {n_qual} of {len(priced)} priced games.")
    a("")
    a("## Dropped prices")
    a(f"- Ties / voids from the NBA log: {ties}.")
    a(f"- Games dropped from the bankroll/edges priced set: {len(dropped)}.")
    reasons = Counter(d["reason"] for d in dropped)
    for reason, n in reasons.most_common():
        a(f"- `{reason}`: {n}")
    a("")
    a("## Caveats")
    a("- **Injury:** `stack_global` is not injury-adjusted.")
    a("- Candles have no size; fills are optimistic for big stakes.")
    a("- Away taker fill is implied from the home YES bid (away candles were not in the pull).")
    a("- The $0.02 fee is Tim's working estimate, not a verified Kalshi fee.")
    a("- Bankroll charts follow Tim's season rule via `Backtest.BankrollPlot`.")
    # verdict
    flat = roi["flat100"]
    edge_ok = (
        flat["roi"] is not None and flat["lo"] is not None and flat["lo"] > 0 and paths["flat100"]["n_bets"] > 0
    )
    if edge_ok:
        a("- Verdict: the 90% CI on flat100 ROI is above zero on this sample; still treat as research, not a live bet, until Tim says otherwise.")
    else:
        a("- **Verdict: PASS.** No validated out-of-sample edge after Tim's $0.02 fee estimate. Ending bankrolls and CIs do not support a BET.")
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def edge_qual(g):
    side, edge, _p, _m = ChosenSide(g["stack_global_home_prob"], g["market_prob_devig_home"])
    return edge >= EDGE_CUT


def GitHash(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), text=True).strip()


def Main() -> int:
    t0 = time.time()
    PreparePath()
    from Matchups import Cpu
    Cpu.LimitThreads()
    Cpu.LowerPriority()
    BP = LoadBankrollPlot()
    hash_main = GitHash(MAIN)
    hash_wt = GitHash(WORKTREE)
    print(f"main={hash_main} worktree={hash_wt} python={sys.version.split()[0]}", flush=True)

    games = LoadSeasonGames()
    print(f"oos games from season files: {len(games)}", flush=True)
    sched_index = LoadSchedIndex()
    market_index = LoadMarketIndex()
    priced_raw, dropped, ties = AttachPrices(games, sched_index, market_index)
    print(f"priced before model join: {len(priced_raw)} dropped={len(dropped)} ties={ties}", flush=True)

    stacked = []
    def prog(msg):
        print(msg, flush=True)

    # One walk per official three-season window, then keep OOS rows.
    seen = set()
    for label, seasons in WALK_WINDOWS.items():
        rows = StackWindow(seasons, progress=prog)
        for r in rows:
            if r["game_id"] in seen:
                continue
            # Keep if this game belongs to this label's OOS season file year.
            stacked.append(r)
            seen.add(r["game_id"])

    merged, missing_model = MergeForecasts(priced_raw, stacked)
    for g in missing_model:
        dropped.append({"game_id": g["game_id"], "date": g.get("game_date"),
                        "event": f"{g.get('away_team')} @ {g.get('home_team')}",
                        "reason": "no_model_forecast"})
    priced = [g for g in merged if g.get("priced") and "stack_global_home_prob" in g]
    print(f"priced with model: {len(priced)} missing_model={len(missing_model)}", flush=True)

    asof = {}
    for g in games:
        asof[g["season_label"]] = g["game_date"]
        asof.setdefault("pooled_oos", g["game_date"])
        asof["pooled_oos"] = max(asof["pooled_oos"], g["game_date"])

    cal, summary = CalibrationRows(priced, asof)
    cal_fields = ["date", "sport", "series", "model", "season", "split",
                  "bin_lo", "bin_hi", "n", "mean_pred", "obs_rate"]
    sum_fields = ["date", "sport", "series", "model", "season", "split",
                  "n", "accuracy", "brier", "log_loss", "ece"]
    WriteCsv(CAL_DIR / f"{CAL_BASE}.csv", cal, cal_fields)
    WriteCsv(CAL_DIR / f"{CAL_BASE}_summary.csv", summary, sum_fields)
    WriteCalibrationMd(CAL_DIR / f"{CAL_BASE}.md", summary, priced, dropped, ties, "", hash_main, hash_wt)

    edges = EdgeRows(priced)
    edge_by_id = {r["event_id"]: r for r in edges}
    edge_fields = ["date", "start_time", "sport", "season", "split", "event_id", "event", "side",
                   "model", "model_prob", "market_prob_devig", "fill_price", "price_basis",
                   "price_time", "fee", "fee_basis", "edge", "edge_vs_fill", "qualifies",
                   "result", "pnl_at_flat_stake"]
    WriteCsv(BANK_DIR / f"{BANK_BASE}_edges.csv", edges, edge_fields)

    global _DAY_SEASON
    _DAY_SEASON = {}
    day_set = []
    for g in games:
        d = g["game_date"]
        if d not in _DAY_SEASON:
            _DAY_SEASON[d] = g["season_label"]
            day_set.append(d)
    all_days = sorted(day_set)
    paths = BankrollPaths(priced, all_days, edge_by_id)
    # Apply season-axis helper: shared game-day index, summer skipped, x starts at 1.
    bank_rows = []
    for name, payload in paths.items():
        dates = [r["date"] for r in payload["rows"]]
        equity = [r["equity"] for r in payload["rows"]]
        frame = BP.FrameBankroll(dates, equity, series=name, mode="index")
        xmap = {row["date"]: int(row["x"]) + 1 for row in frame["rows"]}
        ddmap = {row["date"]: row["drawdown_pct"] for row in frame["rows"]}
        for r in payload["rows"]:
            r["x"] = xmap[r["date"]]
            r["drawdown_pct"] = ddmap[r["date"]]
            bank_rows.append(r)
    bank_rows.sort(key=lambda r: (r["x"], r["strategy"]))
    bank_fields = ["date", "sport", "series", "strategy", "price_basis", "season", "split",
                   "x", "bets_day", "bets_cum", "equity", "drawdown_pct"]
    WriteCsv(BANK_DIR / f"{BANK_BASE}.csv", bank_rows, bank_fields)

    roi = {name: BootstrapRoi(paths[name]["stakes_pnls"]) for name in paths}
    WriteBankrollMd(BANK_DIR / f"{BANK_BASE}.md", paths, roi, priced, dropped, ties, all_days, hash_main, hash_wt)

    # Draft bankroll PNG if matplotlib is available.
    try:
        d, e, s = [], [], []
        for name in ("flat100", "qkelly"):
            for r in paths[name]["rows"]:
                d.append(r["date"]); e.append(r["equity"]); s.append(name)
        frame = BP.FrameBankroll(d, e, s, mode="index")
        BP.PlotBankroll(frame, path=str(BANK_DIR / f"{BANK_BASE}_draft.png"),
                        title="NBA $10k bankroll, stack_global at Kalshi T-60 (summer skipped)",
                        ylabel="Simulated bankroll (US dollars)")
        print("wrote draft png", flush=True)
    except Exception as exc:
        print(f"draft plot skipped: {exc}", flush=True)

    # Number dump for the parent report
    dump = {
        "hash_main": hash_main, "hash_wt": hash_wt,
        "n_season_games": len(games), "n_priced": len(priced),
        "n_dropped": len(dropped), "ties": ties,
        "n_holdout": sum(1 for g in priced if g["split"] == "holdout"),
        "n_forward": sum(1 for g in priced if g["split"] == "forward"),
        "n_qual": sum(1 for g in priced if edge_qual(g)),
        "summary": summary,
        "bankroll": {k: {"end": v["end"], "n_bets": v["n_bets"], "max_dd": v["max_dd"],
                         "ruined": v["ruined"], "ruin_day": v["ruin_day"], "roi": roi[k]}
                     for k, v in paths.items()},
        "drop_reasons": dict(Counter(d["reason"] for d in dropped)),
        "elapsed_s": time.time() - t0,
    }
    (BANK_DIR / f"{BANK_BASE}_run.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: dump[k] for k in dump if k != "summary"}, indent=2, default=str), flush=True)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(Main())
    except Exception:
        import traceback
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        sys.stderr.flush()
        Path(r"E:\AgentWorkspaces\Basketball\basketball\logs\2026-11-13_export-ops-vision.exc").write_text(tb, encoding="utf-8")
        raise
