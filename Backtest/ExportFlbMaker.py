"""Pre-registered favourite-longshot bias and maker replay (NBA, Kalshi T-60).

Fit the FLB rule on games through 2026-01-31 only, freeze and hash the spec,
then score the holdout (2026-02-01 through the end of 2025-26, plus 2026-27
to date). Maker replay has no fitted parameters;  the tune window is
descriptive and the holdout is the test.

Tidy CSV columns match Basebot's 2026-11-13_flb-maker output exactly.
Placement is the earliest valid 1-min candle before T-60 (not T-120). The
t120_* order columns keep Basebot's names and hold that placement quote.

.venv\\Scripts\\python.exe Backtest/ExportFlbMaker.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

MAIN = Path(r"E:\Basketball\basketball")
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from Backtest.ExportOverreactionFade import (  # noqa: E402
    CanonicalHash,
    GitHash,
    LimitCpu,
    LoadBankrollPlot,
    LoadPricedEdges,
    LoadSchedAndMarkets,
    Mid,
    QuoteFromCandle,
    SeasonOf,
    SplitOf,
    ValidCandle,
    WriteCsv,
)

EDGES_IN = Path(r"E:\AgentDocs\Basketbot\2026-11-13_bankroll-10k\2026-11-13_bankroll-10k_edges.csv")
KALSHI_CANDLES = MAIN / "data" / "inplay" / "kalshi" / "candles"
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_flb-maker")
BASE = "2026-11-13_flb-maker"
FEE = 0.02
FEE_BASIS = "tim_est_0.02"
PRICE_BASIS = "kalshi_T60"
EDGE_CUT = 0.03
START_BANK = 10000.0
FLAT_STAKE = 100.0
QKELLY_FRAC = 0.25
QKELLY_CAP = 0.02
T60_SEC = 3600
STALE_SEC = 1800
TUNE_END = "2026-01-31"
HOLDOUT_START = "2026-02-01"
BOOT_REPS = 5000
BOOT_SEED = 20261113
MIN_N_ELIGIBLE = 100
MIN_N_FROZEN_BIN = 20
ET = ZoneInfo("America/New_York")

GEQ_XS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
LEQ_XS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
BIN_EDGES = [(i / 20.0, (i + 1) / 20.0) for i in range(20)]

FLB_BINS_FIELDS = [
    "season", "split", "bin_lo", "bin_hi", "n", "n_days", "mean_price",
    "fair_value", "mean_edge", "edge_ci95_lo", "edge_ci95_hi", "roi",
    "roi_ci95_lo", "roi_ci95_hi", "mean_kalshi_fee", "mean_edge_kalshi_fee",
]
FLB_GRID_FIELDS = ["x", "season", "n", "mean_edge", "roi", "eligible"]
FLB_RATES_FIELDS = ["bin_lo", "bin_hi", "fit_season", "frozen_win_rate"]
ADVERSE_FIELDS = [
    "strategy", "season", "fill_rule", "filled", "n", "win_rate", "mean_price",
    "mean_edge_if_filled_at_limit", "edge_ci95_lo", "edge_ci95_hi",
    "roi_if_filled_at_limit",
]
STRAT_FIELDS = [
    "strategy", "season", "split", "n", "wins", "n_days", "mean_price",
    "fair_value", "mean_pred_edge", "mean_edge", "edge_ci95_lo", "edge_ci95_hi",
    "roi", "roi_ci95_lo", "roi_ci95_hi", "mean_kalshi_fee",
    "mean_edge_kalshi_fee", "edge_kfee_ci95_lo", "edge_kfee_ci95_hi",
    "clears_3c", "fill_rule", "orders", "fill_rate", "note",
]
ORDER_FIELDS = [
    "strategy", "season", "date", "start_time", "event_id", "side",
    "t120_bid", "t120_ask", "limit_c", "price", "cost", "prob", "kfee",
    "window_candles", "filled_at", "filled_through", "result",
]
BANKROLL_FIELDS = [
    "date", "sport", "series", "strategy", "price_basis", "season", "split",
    "x", "bets_day", "bets_cum", "equity", "drawdown_pct",
]
METRICS_FIELDS = [
    "strategy", "scope", "bets", "wins", "staked", "pnl", "start_equity",
    "end_equity", "max_drawdown_pct", "roi",
    "roi_ci90_event_lo", "roi_ci90_event_hi", "roi_ci90_day_lo", "roi_ci90_day_hi",
    "roi_ci90_widest_lo", "roi_ci90_widest_hi", "game_days", "ruin_x",
]

MAKER_NOTE = (
    "optimistic fills (1-min candles; window starts ~1-2h before T-60, "
    "not T-120; away book implied from home YES; no size/queue)"
)


def TakerFee(p: float) -> float:
    return 0.07 * p * (1.0 - p)


def MakerFee(p: float) -> float:
    return 0.0175 * p * (1.0 - p)


def BinOf(ask: float):
    if ask >= 1.0:
        return (0.95, 1.0)
    if ask < 0.0:
        return (0.0, 0.05)
    i = int(math.floor(ask / 0.05 + 1e-12))
    i = min(19, max(0, i))
    return (round(i * 0.05, 2), round((i + 1) * 0.05, 2))


def Mid1Limit(mid: float) -> float:
    cents = int(math.floor(mid * 100.0 - 1.0))
    cents = max(1, min(99, cents))
    return cents / 100.0


def TradeRange(c) -> tuple:
    lo, hi, last, vol = c.get("tr_lo"), c.get("tr_hi"), c.get("last"), c.get("vol")
    traded = False
    if vol is not None:
        try:
            traded = float(vol) > 0.0
        except (TypeError, ValueError):
            traded = False
    else:
        traded = lo is not None or last is not None
    if not traded:
        return False, None, None
    if lo is not None and hi is not None:
        return True, float(lo), float(hi)
    if last is not None:
        v = float(last)
        return True, v, v
    return False, None, None


def WindowFilled(later, side: str, limit: float, through: bool) -> bool:
    for c in later:
        has, lo, hi = TradeRange(c)
        if not has:
            continue
        if side == "home":
            if through:
                if lo < limit:
                    return True
            elif lo <= limit:
                return True
        else:
            thresh = 1.0 - limit
            if through:
                if hi > thresh:
                    return True
            elif hi >= thresh:
                return True
    return False


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
    """triples: (day, cost, pnl). CI for sum(pnl)/sum(cost)."""
    if not triples:
        return {"roi": None, "lo": None, "hi": None, "n": 0, "n_days": 0}
    costs = [c for _d, c, _p in triples]
    pnls = [p for _d, _c, p in triples]
    tot_c = sum(costs)
    point = (sum(pnls) / tot_c) if tot_c else None
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


def BootstrapEventRoi(stakes_pnls, reps=BOOT_REPS, seed=BOOT_SEED, level=0.90):
    if not stakes_pnls:
        return {"roi": None, "lo": None, "hi": None, "n": 0}
    stakes = [s for s, _p in stakes_pnls]
    pnls = [p for _s, p in stakes_pnls]
    tot = sum(stakes)
    point = (sum(pnls) / tot) if tot else None
    rng = random.Random(seed)
    n = len(pnls)
    boots = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        s = sum(stakes[i] for i in idx)
        if s > 0:
            boots.append(sum(pnls[i] for i in idx) / s)
    boots.sort()
    if not boots:
        return {"roi": point, "lo": None, "hi": None, "n": n}
    alpha = (1.0 - level) / 2.0
    return {
        "roi": point,
        "lo": boots[int(alpha * len(boots))],
        "hi": boots[min(len(boots) - 1, int((1.0 - alpha) * len(boots)))],
        "n": n,
    }


def Mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def Fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def FmtSigned(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}"


def FmtCI(lo, hi, nd=4):
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:+.{nd}f}, {hi:+.{nd}f}]"


def FairDollars(p):
    if p is None:
        return "n/a"
    return f"${p:.2f}"


def AppendTrial(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(rec)
    rec.setdefault("ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def WritePrereg(path: Path, repo_hash: str) -> None:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    text = f"""# {BASE} - PRE-REGISTRATION (written before any FLB / maker result was computed)

Frozen: {now} (ET). Repo HEAD {repo_hash}. Research simulation only; no bets.

## Game set (fixed)
- The 2026-11-13 calibration / bankroll-10k kalshi_T60 priced set (max quote age 30 min at T-60): 1,469 games. No re-selection.
- **Fit/tune split = games through {TUNE_END} inclusive. Holdout = {HOLDOUT_START} through the end of 2025-26, plus 2026-27 to date.** The holdout is touched once with the frozen rules below.

## Common rules
- Price paid = taker: side yes_ask at T-60; maker: the limit price. Cost = price paid + $0.02 (Tim's working estimate, unverified; fee_basis {FEE_BASIS}). Kalshi fee shown as a second column: taker 0.07*p*(1-p), maker 0.0175*p*(1-p) per contract (approximate published schedule; maker fees apply only on some markets - verify). p is the price paid.
- edge = prob - cost for predicted edge. Realised edge per $1 contract = result - cost. Value only at predicted edge >= 0.03 (3c).
- Holdout headline per strategy: N bets, mean realised edge per $1 contract, 95% DAY-BLOCK bootstrap CI (resample holdout game days, {BOOT_REPS} reps, seed {BOOT_SEED}). Also ROI per $ staked (sum(result-cost)/sum(cost)) with the same CI.
- **Verdict rule: a strategy "clears 3c" only if its holdout mean realised edge >= 0.03 AND the 95% day-block CI lower bound > 0.** Multiple variants are tested; this is noted, no correction applied beyond reporting all of them. Default call is PASS unless at least one pre-registered headline strategy clears 3c.
- $10k bankroll on the holdout only (one continuous bankroll, no reset): flat100 ($100/bet, contracts = 100/cost) and qkelly (0.25*(p-c)/(1-c) of start-of-day equity, cap 2%). p = the strategy's frozen probability. Charts: holdout game-day index, first to last game day, season marker, no offseason.

## (a) Favourite-longshot bias
- Unit = each SIDE of each priced game; price = side yes_ask at T-60; bins of width 0.05 on the ask. Away ask = 1 - home bid (only home YES candles were pulled).
- Per bin per season/split: N sides, mean ask, realised win rate, ROI per $ at cost (ask + 0.02), fair value = win rate in $0.xx.
- Two families, at most one side per game (ties skipped):
  - "buy the higher-ask side if ask >= x", grid x in {{0.50, 0.55, ..., 0.90}}
  - "buy the lower-ask side if ask <= x", grid x in {{0.10, 0.15, ..., 0.50}}
- Choose (family, x) on TUNE ONLY = max tune mean realised edge among grid points with >= {MIN_N_ELIGIBLE} tune bets. Ties: prefer >= (favourites), then larger N, then smaller x. If no eligible point, freeze "no bet". All grid points are logged in the trial registry.
- Frozen probability for edge/Kelly = tune realised win rate of the side's 0.05 bin (empty or N<{MIN_N_FROZEN_BIN} bin -> no bet on the 3c rule).
  Bet on the holdout only if the frozen rule matches AND frozen bin rate - (ask + 0.02) >= 0.03.
  Also report the plain rule (frozen family and x, no 3c filter) as a secondary line.

## (b) Maker replay
- Order placed at the earliest valid 1-min candle at or before T-60 (the start of the pulled window, typically 1-2 hours before T-60; this is NOT T-120 and is not the listed market open). Quote at placement = that candle's yes_bid / yes_ask. Limits: L_bid = bid; L_mid1 = floor(mid_cents - 1) cents (may sit at/below the bid).
- Fill (primary, "at_or_through"): a later 1-min candle with t strictly after placement and t at or before scheduled tip has a trade low <= L (home) or trade high >= 1-L (away, implied from the home YES book). Strict sensitivity: trade low < L / trade high > 1-L ("through"). Queue position, size and partial fills are unknown -> fills are OPTIMISTIC for "at"; labelled so. If bid/ask history is missing at placement the game is skipped for maker strategies. Fills inferred from the home book for the away side are also labelled optimistic.
- Strategies:
  (i) market favourite control: side with the higher de-vigged mid at the decision time (T-60 mid for taker; placement mid for maker); every game (mid ties skipped); prob = de-vigged mid.
  (ii) taker-at-ask of that favourite, compared with maker_bid and maker_mid1 (and strict through variants).
- No parameters are fitted in (b); tune is reported as descriptive/in-sample, holdout is the test.
- Adverse selection: for maker orders, win rate and hypothetical ROI at the limit for FILLED vs UNFILLED orders.
- **Candle-coverage caveat: the pulled 1-min candles cover only about 1-2 hours before T-60, plus any pre-tip bars after T-60. This is a late-pregame window, not open-to-tip and not T-120.**

## Contamination
- (a) and (b)(i) use no model probability and are clean with respect to the holdout, provided the rule is frozen on tune only. stack_global is not used in this job.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def AttachCandles(games, sched, ticker_by_key):
    coverage = []
    for g in games:
        key = (g["game_date"], g["home"], g["away"])
        sch = sched.get(key, {})
        ticker = sch.get("ticker") or ticker_by_key.get(key)
        sched_ts = sch.get("sched_ts")
        g["ticker"] = ticker
        g["sched_ts"] = sched_ts
        g["home_ask"] = g["home_ask_t60_edge"]
        g["away_ask"] = g["away_ask_t60_edge"]
        g["home_mid"] = g["p_mkt"]
        g["away_mid"] = 1.0 - g["p_mkt"]
        g["has_place"] = False
        g["place"] = None
        g["later_candles"] = []
        g["drop_maker"] = None
        g["ba_missing"] = False
        if not ticker or not sched_ts:
            g["drop_maker"] = "no_ticker_or_sched"
            continue
        path = KALSHI_CANDLES / f"{ticker}.json"
        if not path.exists():
            g["drop_maker"] = "no_candle_file"
            g["ba_missing"] = True
            continue
        raw = json.loads(path.read_text(encoding="utf-8")).get("candles") or []
        target = int(sched_ts) - T60_SEC
        valid_before = [c for c in raw if ValidCandle(c) and int(c["t"]) <= target]
        if not valid_before:
            g["drop_maker"] = "no_quote_at_or_before_T-60"
            g["ba_missing"] = True
            continue
        t60c = valid_before[-1]
        age = target - int(t60c["t"])
        if age > STALE_SEC:
            g["drop_maker"] = "stale_t60_>30min"
            continue
        place = QuoteFromCandle(valid_before[0])
        later = [c for c in raw if int(c["t"]) > int(place["t"]) and int(c["t"]) <= int(sched_ts)]
        t60 = QuoteFromCandle(t60c)
        g["has_place"] = True
        g["place"] = place
        g["t60_quote"] = t60
        g["later_candles"] = later
        g["place_to_t60_s"] = target - int(place["t"])
        g["n_later"] = len(later)
        g["n_later_with_trade"] = sum(1 for c in later if TradeRange(c)[0])
        post_t60 = [c for c in later if int(c["t"]) > target]
        g["n_post_t60_pre_tip"] = len(post_t60)
        # Recompute T-60 book from the candle when present (should match the priced set).
        g["home_bid"] = t60["bid"]
        g["home_ask"] = t60["ask"]
        g["home_mid"] = t60["mid"]
        g["away_bid"] = 1.0 - t60["ask"]
        g["away_ask"] = 1.0 - t60["bid"]
        g["away_mid"] = 1.0 - t60["mid"]
        coverage.append({
            "game_id": g["game_id"],
            "date": g["game_date"],
            "place_to_t60_s": g["place_to_t60_s"],
            "n_later": g["n_later"],
            "n_later_with_trade": g["n_later_with_trade"],
            "n_post_t60_pre_tip": g["n_post_t60_pre_tip"],
        })
    return coverage


def SideRows(games):
    rows = []
    for g in games:
        for side, ask, won, mid in (
            ("home", float(g["home_ask"]), int(g["home_win"]), float(g["home_mid"])),
            ("away", float(g["away_ask"]), 1 - int(g["home_win"]), float(g["away_mid"])),
        ):
            cost = ask + FEE
            pnl = (1.0 if won else 0.0) - cost
            kfee = TakerFee(ask)
            rows.append({
                "game_id": g["game_id"],
                "date": g["game_date"],
                "start_time": g["start_time"],
                "season": SeasonOf(g["game_date"]),
                "split": SplitOf(g["game_date"]),
                "side": side,
                "ask": ask,
                "mid": mid,
                "won": won,
                "cost": cost,
                "pnl": pnl,
                "kfee": kfee,
                "pnl_kfee": (1.0 if won else 0.0) - (ask + kfee),
                "bin": BinOf(ask),
            })
    return rows


def SummarizeBin(sides):
    if not sides:
        return {
            "n": 0, "n_days": 0, "mean_price": None, "fair_value": None,
            "mean_edge": None, "edge_ci95_lo": None, "edge_ci95_hi": None,
            "roi": None, "roi_ci95_lo": None, "roi_ci95_hi": None,
            "mean_kalshi_fee": None, "mean_edge_kalshi_fee": None,
        }
    edge = BootstrapDayMean([(s["date"], s["pnl"]) for s in sides])
    roi = BootstrapDayRoi([(s["date"], s["cost"], s["pnl"]) for s in sides])
    return {
        "n": len(sides),
        "n_days": edge["n_days"],
        "mean_price": Mean([s["ask"] for s in sides]),
        "fair_value": Mean([s["won"] for s in sides]),
        "mean_edge": edge["mean"],
        "edge_ci95_lo": edge["lo"],
        "edge_ci95_hi": edge["hi"],
        "roi": roi["roi"],
        "roi_ci95_lo": roi["lo"],
        "roi_ci95_hi": roi["hi"],
        "mean_kalshi_fee": Mean([s["kfee"] for s in sides]),
        "mean_edge_kalshi_fee": Mean([s["pnl_kfee"] for s in sides]),
    }


def BuildBinTable(sides, season, split):
    rows = []
    for lo, hi in BIN_EDGES:
        group = [s for s in sides if s["bin"] == (lo, hi)]
        rec = SummarizeBin(group)
        rec.update({"season": season, "split": split, "bin_lo": lo, "bin_hi": hi})
        rows.append(rec)
    return rows


def PickSides(sides_by_game, op, x):
    picks = []
    for _gid, sides in sides_by_game.items():
        cands = []
        for s in sides:
            if op == "geq" and s["ask"] >= x - 1e-12:
                cands.append(s)
            elif op == "leq" and s["ask"] <= x + 1e-12:
                cands.append(s)
        if not cands:
            continue
        if len(cands) == 2 and abs(cands[0]["ask"] - cands[1]["ask"]) < 1e-12:
            continue
        picks.append(max(cands, key=lambda s: s["ask"]) if op == "geq" else min(cands, key=lambda s: s["ask"]))
    return picks


def GridPoint(sides_by_game, op, x, season_label):
    picks = PickSides(sides_by_game, op, x)
    summ = SummarizeBin(picks)
    return {
        "x": x,
        "season": season_label,
        "n": summ["n"],
        "mean_edge": summ["mean_edge"],
        "roi": summ["roi"],
        "eligible": bool(summ["n"] >= MIN_N_ELIGIBLE),
        "op": op,
        "picks": picks,
        "summ": summ,
    }


def ChooseRule(grid_rows):
    eligible = [r for r in grid_rows if r["eligible"] and r["mean_edge"] is not None]
    if not eligible:
        return None
    # max mean edge; ties: prefer geq, then larger n, then smaller x
    eligible.sort(key=lambda r: (-r["mean_edge"], 0 if r["op"] == "geq" else 1, -r["n"], r["x"]))
    return eligible[0]


def FrozenRates(tune_sides):
    rates = {}
    rows = []
    for lo, hi in BIN_EDGES:
        group = [s for s in tune_sides if s["bin"] == (lo, hi)]
        if len(group) >= MIN_N_FROZEN_BIN:
            rate = Mean([s["won"] for s in group])
            rates[(lo, hi)] = rate
            rows.append({"bin_lo": lo, "bin_hi": hi, "fit_season": "tune", "frozen_win_rate": rate})
        else:
            rows.append({"bin_lo": lo, "bin_hi": hi, "fit_season": "tune", "frozen_win_rate": None})
    return rates, rows


def ApplyFlb(sides_by_game, op, x, rates, require_3c):
    picks = PickSides(sides_by_game, op, x)
    out = []
    for s in picks:
        p = rates.get(s["bin"])
        if require_3c:
            if p is None:
                continue
            if p - s["cost"] < EDGE_CUT:
                continue
        rec = dict(s)
        rec["prob"] = p
        rec["pred_edge"] = (p - s["cost"]) if p is not None else None
        rec["price"] = s["ask"]
        rec["fill_price"] = s["ask"]
        rec["result"] = s["won"]
        rec["kfee"] = TakerFee(s["ask"])
        rec["pnl_kfee"] = (1.0 if s["won"] else 0.0) - (s["ask"] + rec["kfee"])
        out.append(rec)
    return out


def SideQuote(place, side):
    if side == "home":
        return place["bid"], place["ask"], place["mid"]
    return 1.0 - place["ask"], 1.0 - place["bid"], 1.0 - place["mid"]


def FavSide(mid_home):
    if mid_home is None:
        return None
    if mid_home > 0.5 + 1e-12:
        return "home"
    if mid_home < 0.5 - 1e-12:
        return "away"
    return None


def SummarizeTrades(trades, fee_kind="taker"):
    if not trades:
        return {
            "n": 0, "wins": 0, "n_days": 0, "mean_price": None, "fair_value": None,
            "mean_pred_edge": None, "mean_edge": None, "edge_ci95_lo": None,
            "edge_ci95_hi": None, "roi": None, "roi_ci95_lo": None, "roi_ci95_hi": None,
            "mean_kalshi_fee": None, "mean_edge_kalshi_fee": None,
            "edge_kfee_ci95_lo": None, "edge_kfee_ci95_hi": None, "clears_3c": False,
        }
    edge = BootstrapDayMean([(t["date"], t["pnl"]) for t in trades])
    roi = BootstrapDayRoi([(t["date"], t["cost"], t["pnl"]) for t in trades])
    kfee_pairs = [(t["date"], t["pnl_kfee"]) for t in trades]
    kci = BootstrapDayMean(kfee_pairs)
    pred = [t["pred_edge"] for t in trades if t.get("pred_edge") is not None]
    clears = bool(
        edge["mean"] is not None and edge["mean"] >= EDGE_CUT
        and edge["lo"] is not None and edge["lo"] > 0
    )
    return {
        "n": len(trades),
        "wins": int(sum(t["result"] for t in trades)),
        "n_days": edge["n_days"],
        "mean_price": Mean([t["price"] for t in trades]),
        "fair_value": Mean([t["result"] for t in trades]),
        "mean_pred_edge": Mean(pred) if pred else None,
        "mean_edge": edge["mean"],
        "edge_ci95_lo": edge["lo"],
        "edge_ci95_hi": edge["hi"],
        "roi": roi["roi"],
        "roi_ci95_lo": roi["lo"],
        "roi_ci95_hi": roi["hi"],
        "mean_kalshi_fee": Mean([t["kfee"] for t in trades]),
        "mean_edge_kalshi_fee": kci["mean"],
        "edge_kfee_ci95_lo": kci["lo"],
        "edge_kfee_ci95_hi": kci["hi"],
        "clears_3c": clears,
    }


def StratRow(name, season, split, trades, fill_rule="", orders=None, fill_rate=None, note=""):
    rec = SummarizeTrades(trades)
    rec.update({
        "strategy": name,
        "season": season,
        "split": split,
        "fill_rule": fill_rule,
        "orders": orders if orders is not None else (len(trades) if trades else 0),
        "fill_rate": fill_rate,
        "note": note,
    })
    return rec


def Bankroll(trades, all_days, day_season, strategy_tag):
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["date"]].append(t)

    def Run(kind):
        equity = START_BANK
        peak = START_BANK
        bets_cum = 0
        wins = 0
        ruined = False
        ruin_day = None
        rows, stakes_pnls, day_triples = [], [], []
        for i, day in enumerate(all_days):
            start_of_day = equity
            bets_day = 0
            for t in by_day.get(day, []):
                if ruined or equity < 1.0:
                    if not ruined:
                        ruined = True
                        ruin_day = day
                    continue
                price = t["price"]
                result = t["result"]
                c = price + FEE
                if kind == "flat100":
                    stake = FLAT_STAKE
                    if equity < stake:
                        ruined = True
                        ruin_day = day
                        continue
                    pnl = (stake / c) * result - stake
                else:
                    p = t.get("prob")
                    if p is None or c >= 1.0 or c <= 0:
                        continue
                    kelly = (p - c) / (1.0 - c)
                    stake = max(0.0, QKELLY_FRAC * kelly * start_of_day)
                    stake = min(stake, QKELLY_CAP * start_of_day, equity)
                    stake = math.floor(stake * 100.0 + 1e-9) / 100.0
                    if stake < 1.0:
                        continue
                    pnl = (stake / c) * result - stake
                equity += pnl
                bets_day += 1
                bets_cum += 1
                wins += int(result)
                stakes_pnls.append((stake, pnl))
                day_triples.append((day, stake, pnl))
            if equity > peak:
                peak = equity
            dd = 0.0 if peak == 0 else min(0.0, equity / peak - 1.0)
            season = day_season.get(day, SeasonOf(day))
            rows.append({
                "date": day, "sport": "nba",
                "series": f"nba_{kind}_{PRICE_BASIS}_{strategy_tag}",
                "strategy": f"{kind}:{strategy_tag}",
                "price_basis": PRICE_BASIS,
                "season": season, "split": "holdout",
                "x": i + 1,
                "bets_day": bets_day, "bets_cum": bets_cum,
                "equity": equity, "drawdown_pct": dd,
            })
        event90 = BootstrapEventRoi(stakes_pnls, level=0.90)
        day90 = BootstrapDayRoi(day_triples, level=0.90) if day_triples else {"roi": None, "lo": None, "hi": None}
        # Widest = larger interval width (more conservative).
        widest_lo, widest_hi = event90["lo"], event90["hi"]
        if day90["lo"] is not None and event90["lo"] is not None:
            if (day90["hi"] - day90["lo"]) >= (event90["hi"] - event90["lo"]):
                widest_lo, widest_hi = day90["lo"], day90["hi"]
        staked = sum(s for s, _p in stakes_pnls)
        pnl = sum(p for _s, p in stakes_pnls)
        metrics = {
            "strategy": f"{kind}:{strategy_tag}",
            "scope": "holdout",
            "bets": bets_cum,
            "wins": wins,
            "staked": staked,
            "pnl": pnl,
            "start_equity": START_BANK,
            "end_equity": equity,
            "max_drawdown_pct": min((r["drawdown_pct"] for r in rows), default=0.0),
            "roi": (pnl / staked) if staked else None,
            "roi_ci90_event_lo": event90["lo"],
            "roi_ci90_event_hi": event90["hi"],
            "roi_ci90_day_lo": day90["lo"],
            "roi_ci90_day_hi": day90["hi"],
            "roi_ci90_widest_lo": widest_lo,
            "roi_ci90_widest_hi": widest_hi,
            "game_days": len(all_days),
            "ruin_x": ruin_day,
        }
        return {"rows": rows, "metrics": metrics, "end": equity, "n_bets": bets_cum}

    return {name: Run(name) for name in ("flat100", "qkelly")}


def MdTable(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def BuildMakerOrders(games, split_filter=None):
    """Resting fav orders at placement bid and mid-1c. Returns order rows (filled or not)."""
    orders = []
    skipped = 0
    for g in games:
        if split_filter and SplitOf(g["game_date"]) != split_filter:
            continue
        if not g.get("has_place"):
            skipped += 1
            continue
        place = g["place"]
        side = FavSide(place["mid"])
        if side is None:
            skipped += 1
            continue
        bid, ask, mid = SideQuote(place, side)
        won = int(g["home_win"]) if side == "home" else 1 - int(g["home_win"])
        later = g.get("later_candles") or []
        limits = {
            "fav_maker_bid": bid,
            "fav_maker_mid1": Mid1Limit(mid),
        }
        season = "tune" if SplitOf(g["game_date"]) == "tune" else "holdout"
        for name, limit in limits.items():
            if limit is None or limit <= 0 or limit >= 1:
                continue
            filled_at = WindowFilled(later, side, limit, through=False)
            filled_through = WindowFilled(later, side, limit, through=True)
            kfee = MakerFee(limit)
            cost = limit + FEE
            orders.append({
                "strategy": name,
                "season": season,
                "date": g["game_date"],
                "start_time": g["start_time"],
                "event_id": g["game_id"],
                "side": side,
                "t120_bid": bid,
                "t120_ask": ask,
                "limit_c": int(round(limit * 100)),
                "price": limit,
                "cost": cost,
                "prob": mid,
                "kfee": kfee,
                "window_candles": len(later),
                "filled_at": filled_at,
                "filled_through": filled_through,
                "result": won,
                "pred_edge": mid - cost,
                "pnl_if": (1.0 if won else 0.0) - cost,
                "pnl_kfee_if": (1.0 if won else 0.0) - (limit + kfee),
                "split": SplitOf(g["game_date"]),
                "game_season": SeasonOf(g["game_date"]),
                "ba_missing": bool(g.get("ba_missing")),
            })
    return orders, skipped


def OrdersToTrades(orders, fill_key):
    trades = []
    for o in orders:
        if not o[fill_key]:
            continue
        trades.append({
            "date": o["date"],
            "price": o["price"],
            "cost": o["cost"],
            "prob": o["prob"],
            "pred_edge": o["pred_edge"],
            "result": o["result"],
            "pnl": o["pnl_if"],
            "kfee": o["kfee"],
            "pnl_kfee": o["pnl_kfee_if"],
            "fill_price": o["price"],
        })
    return trades


def AdverseRows(orders, season):
    rows = []
    for strat in ("fav_maker_bid", "fav_maker_mid1"):
        group = [o for o in orders if o["strategy"] == strat and o["season"] == season]
        for fill_rule, key in (("filled_at", "filled_at"), ("filled_through", "filled_through")):
            for filled in (True, False):
                sub = [o for o in group if bool(o[key]) is filled]
                if not sub:
                    rows.append({
                        "strategy": strat, "season": season, "fill_rule": fill_rule,
                        "filled": filled, "n": 0, "win_rate": None, "mean_price": None,
                        "mean_edge_if_filled_at_limit": None, "edge_ci95_lo": None,
                        "edge_ci95_hi": None, "roi_if_filled_at_limit": None,
                    })
                    continue
                edge = BootstrapDayMean([(o["date"], o["pnl_if"]) for o in sub])
                roi = BootstrapDayRoi([(o["date"], o["cost"], o["pnl_if"]) for o in sub])
                rows.append({
                    "strategy": strat,
                    "season": season,
                    "fill_rule": fill_rule,
                    "filled": filled,
                    "n": len(sub),
                    "win_rate": Mean([o["result"] for o in sub]),
                    "mean_price": Mean([o["price"] for o in sub]),
                    "mean_edge_if_filled_at_limit": edge["mean"],
                    "edge_ci95_lo": edge["lo"],
                    "edge_ci95_hi": edge["hi"],
                    "roi_if_filled_at_limit": roi["roi"],
                })
    return rows


def BuildFavTaker(games, split_filter=None):
    trades = []
    for g in games:
        if split_filter and SplitOf(g["game_date"]) != split_filter:
            continue
        side = FavSide(g["home_mid"])
        if side is None:
            continue
        ask = g["home_ask"] if side == "home" else g["away_ask"]
        mid = g["home_mid"] if side == "home" else g["away_mid"]
        won = int(g["home_win"]) if side == "home" else 1 - int(g["home_win"])
        cost = ask + FEE
        kfee = TakerFee(ask)
        trades.append({
            "date": g["game_date"],
            "price": ask,
            "cost": cost,
            "prob": mid,
            "pred_edge": mid - cost,
            "result": won,
            "pnl": (1.0 if won else 0.0) - cost,
            "kfee": kfee,
            "pnl_kfee": (1.0 if won else 0.0) - (ask + kfee),
            "fill_price": ask,
            "season": SeasonOf(g["game_date"]),
            "split": SplitOf(g["game_date"]),
        })
    return trades


def BinMdRows(bin_rows):
    out = []
    for r in bin_rows:
        if r["n"] == 0:
            continue
        out.append([
            f"{r['bin_lo']:.2f}-{r['bin_hi']:.2f}",
            str(r["n"]),
            FairDollars(r["mean_price"]),
            FairDollars(r["fair_value"]),
            FmtSigned(r["mean_edge"]),
            FmtCI(r["edge_ci95_lo"], r["edge_ci95_hi"]),
            FmtSigned(r["roi"]),
            FmtSigned(r["mean_edge_kalshi_fee"]),
        ])
    return out


def StratMdRows(rows):
    out = []
    for r in rows:
        fr = "-" if r.get("fill_rate") in (None, "") else Fmt(r["fill_rate"], 4)
        out.append([
            r["strategy"],
            str(r["n"]),
            fr,
            FairDollars(r["mean_price"]) if r["n"] else "-",
            FairDollars(r["fair_value"]) if r["n"] else "-",
            FmtSigned(r["mean_pred_edge"]),
            FmtSigned(r["mean_edge"]),
            FmtCI(r["edge_ci95_lo"], r["edge_ci95_hi"]),
            "{0} {1}".format(FmtSigned(r["roi"]), FmtCI(r["roi_ci95_lo"], r["roi_ci95_hi"])),
            "{0} {1}".format(FmtSigned(r["mean_edge_kalshi_fee"]), FmtCI(r["edge_kfee_ci95_lo"], r["edge_kfee_ci95_hi"])),
            "yes" if r["clears_3c"] else "no",
        ])
    return out


def Main() -> int:
    LimitCpu()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials_path = OUT_DIR / f"{BASE}_trials.jsonl"
    repo_hash = GitHash(MAIN)

    # 1. Pre-register the protocol before any result is computed.
    prereg_path = OUT_DIR / f"{BASE}_preregistration.md"
    WritePrereg(prereg_path, repo_hash)
    AppendTrial(trials_path, {
        "phase": "preregistration",
        "path": str(prereg_path),
        "repo_head": repo_hash,
        "tune_end": TUNE_END,
        "holdout_start": HOLDOUT_START,
        "boot_reps": BOOT_REPS,
        "boot_seed": BOOT_SEED,
        "min_n_eligible": MIN_N_ELIGIBLE,
        "min_n_frozen_bin": MIN_N_FROZEN_BIN,
        "fee": FEE,
        "fee_basis": FEE_BASIS,
    })

    games = LoadPricedEdges()
    sched, _open_by_ticker, ticker_by_key = LoadSchedAndMarkets()
    coverage = AttachCandles(games, sched, ticker_by_key)

    all_sides = SideRows(games)
    tune_games = [g for g in games if SplitOf(g["game_date"]) == "tune"]
    hold_games = [g for g in games if SplitOf(g["game_date"]) == "holdout"]
    tune_sides = [s for s in all_sides if s["split"] == "tune"]
    hold_sides = [s for s in all_sides if s["split"] == "holdout"]

    def ByGame(sides):
        d = defaultdict(list)
        for s in sides:
            d[s["game_id"]].append(s)
        return d

    tune_by = ByGame(tune_sides)
    hold_by = ByGame(hold_sides)

    # 2. Fit on TUNE ONLY.
    bins_tune = BuildBinTable(tune_sides, "2025-26", "tune")
    grid_geq = [GridPoint(tune_by, "geq", x, "tune") for x in GEQ_XS]
    grid_leq = [GridPoint(tune_by, "leq", x, "tune") for x in LEQ_XS]
    chosen = ChooseRule(grid_geq + grid_leq)
    rates, rate_rows = FrozenRates(tune_sides)

    if chosen is None:
        frozen = {
            "rule": "no_bet",
            "op": None,
            "x": None,
            "tune_n": 0,
            "tune_mean_edge": None,
            "tune_roi": None,
            "reason": "no eligible grid point with n>={0}".format(MIN_N_ELIGIBLE),
        }
    else:
        frozen = {
            "rule": "buy_if_ask_{0}_x".format(chosen["op"]),
            "op": chosen["op"],
            "x": chosen["x"],
            "tune_n": chosen["n"],
            "tune_mean_edge": chosen["mean_edge"],
            "tune_roi": chosen["roi"],
            "reason": "max tune mean realised edge among eligible (n>={0})".format(MIN_N_ELIGIBLE),
        }

    frozen_spec = {
        "job": "flb-maker",
        "sport": "nba",
        "price_basis": PRICE_BASIS,
        "fee": FEE,
        "fee_basis": FEE_BASIS,
        "tune_end": TUNE_END,
        "holdout_start": HOLDOUT_START,
        "n_games_total": len(games),
        "n_games_tune": len(tune_games),
        "n_games_holdout": len(hold_games),
        "flb": frozen,
        "frozen_bin_min_n": MIN_N_FROZEN_BIN,
        "frozen_bin_rates": [
            {"bin_lo": r["bin_lo"], "bin_hi": r["bin_hi"], "frozen_win_rate": r["frozen_win_rate"]}
            for r in rate_rows if r["frozen_win_rate"] is not None
        ],
        "grid_geq_tune": [
            {"x": r["x"], "n": r["n"], "mean_edge": r["mean_edge"], "roi": r["roi"], "eligible": r["eligible"]}
            for r in grid_geq
        ],
        "grid_leq_tune": [
            {"x": r["x"], "n": r["n"], "mean_edge": r["mean_edge"], "roi": r["roi"], "eligible": r["eligible"]}
            for r in grid_leq
        ],
        "maker": {
            "placement": "earliest valid 1-min candle at or before T-60",
            "limits": ["bid", "mid_minus_1c"],
            "fill_primary": "at_or_through",
            "fill_strict": "through",
            "control": "market_fav",
        },
        "bootstrap": {"reps": BOOT_REPS, "seed": BOOT_SEED, "level": 0.95, "cluster": "day"},
        "repo_head_at_freeze": repo_hash,
    }
    frozen_hash = CanonicalHash(frozen_spec)
    frozen_spec["sha256"] = frozen_hash
    spec_path = OUT_DIR / f"{BASE}_spec.json"
    spec_path.write_text(json.dumps(frozen_spec, indent=2, default=str) + "\n", encoding="utf-8")
    (OUT_DIR / f"{BASE}_spec.sha256").write_text(frozen_hash + "\n", encoding="utf-8")

    # Log tune grid BEFORE holdout scoring.
    for r in grid_geq + grid_leq:
        AppendTrial(trials_path, {
            "phase": "tune_grid",
            "op": r["op"],
            "x": r["x"],
            "n": r["n"],
            "mean_edge": r["mean_edge"],
            "roi": r["roi"],
            "eligible": r["eligible"],
            "frozen_hash": frozen_hash,
        })
    AppendTrial(trials_path, {
        "phase": "freeze",
        "sha256": frozen_hash,
        "spec_path": str(spec_path),
        "chosen": frozen,
    })

    # 3. Holdout is scored only after the hash is on disk.
    bins_h_2526 = BuildBinTable([s for s in hold_sides if s["season"] == "2025-26"], "2025-26", "holdout")
    bins_h_2627 = BuildBinTable([s for s in hold_sides if s["season"] == "2026-27"], "2026-27", "holdout")
    bins_h_pool = BuildBinTable(hold_sides, "holdout", "holdout")
    bin_rows = bins_tune + bins_h_2526 + bins_h_2627 + bins_h_pool

    grid_geq_h = [GridPoint(hold_by, "geq", x, "holdout") for x in GEQ_XS]
    grid_leq_h = [GridPoint(hold_by, "leq", x, "holdout") for x in LEQ_XS]

    # Selected-direction grid file (Basebot columns). Alt family in flb_grid_alt.csv.
    if frozen["op"] == "leq":
        grid_main_tune, grid_alt_tune = grid_leq, grid_geq
        grid_main_h, grid_alt_h = grid_leq_h, grid_geq_h
        main_op, alt_op = "leq", "geq"
    else:
        grid_main_tune, grid_alt_tune = grid_geq, grid_leq
        grid_main_h, grid_alt_h = grid_geq_h, grid_leq_h
        main_op, alt_op = (frozen["op"] or "geq"), ("leq" if frozen["op"] != "leq" else "geq")
        if frozen["op"] is None:
            main_op, alt_op = "geq", "leq"
            grid_main_tune, grid_alt_tune = grid_geq, grid_leq
            grid_main_h, grid_alt_h = grid_geq_h, grid_leq_h

    def GridCsv(rows, season):
        return [{
            "x": r["x"], "season": season, "n": r["n"],
            "mean_edge": r["mean_edge"], "roi": r["roi"], "eligible": r["eligible"],
        } for r in rows]

    # FLB strategy bets
    flb_hold_3c, flb_hold_plain = [], []
    flb_tune_3c, flb_tune_plain = [], []
    if frozen["op"] is not None:
        flb_hold_3c = ApplyFlb(hold_by, frozen["op"], frozen["x"], rates, True)
        flb_hold_plain = ApplyFlb(hold_by, frozen["op"], frozen["x"], rates, False)
        flb_tune_3c = ApplyFlb(tune_by, frozen["op"], frozen["x"], rates, True)
        flb_tune_plain = ApplyFlb(tune_by, frozen["op"], frozen["x"], rates, False)

    fav_taker_h = BuildFavTaker(games, "holdout")
    fav_taker_t = BuildFavTaker(games, "tune")

    maker_orders, maker_skipped = BuildMakerOrders(games)
    maker_h = [o for o in maker_orders if o["season"] == "holdout"]
    maker_t = [o for o in maker_orders if o["season"] == "tune"]

    strat_rows = []
    def AddStrats(split, flb3, flbp, fav, makers):
        season = "tune" if split == "tune" else "holdout"
        strat_rows.append(StratRow("flb_rule_3c", season, split, flb3, note="taker at T-60 ask; frozen rule + 3c overlay"))
        strat_rows.append(StratRow("flb_rule_plain", season, split, flbp, note="taker at T-60 ask; frozen rule, no 3c filter"))
        strat_rows.append(StratRow("fav_taker", season, split, fav, note="taker-at-ask market favourite at T-60"))
        for name, fill_key, fill_rule, label in (
            ("fav_maker_bid", "filled_at", "at_or_through", "maker at placement bid"),
            ("fav_maker_bid_strict", "filled_through", "through", "maker at placement bid, through only"),
            ("fav_maker_mid1", "filled_at", "at_or_through", "maker at placement mid-1c"),
            ("fav_maker_mid1_strict", "filled_through", "through", "maker at placement mid-1c, through only"),
        ):
            base = name.replace("_strict", "")
            pool = [o for o in makers if o["strategy"] == base]
            trades = OrdersToTrades(pool, fill_key)
            fill_rate = (len(trades) / len(pool)) if pool else None
            strat_rows.append(StratRow(
                name, season, split, trades,
                fill_rule=fill_rule, orders=len(pool), fill_rate=fill_rate,
                note=MAKER_NOTE + "; " + label,
            ))

    AddStrats("tune", flb_tune_3c, flb_tune_plain, fav_taker_t, maker_t)
    AddStrats("holdout", flb_hold_3c, flb_hold_plain, fav_taker_h, maker_h)

    adverse = AdverseRows(maker_orders, "tune") + AdverseRows(maker_orders, "holdout")

    # Bankrolls on holdout
    hold_days = sorted({g["game_date"] for g in hold_games})
    day_season = {d: SeasonOf(d) for d in hold_days}
    bank_rows, bank_metrics = [], []
    bank_map = {}
    for tag, trades in (
        ("flb_rule_3c", flb_hold_3c),
        ("flb_rule_plain", flb_hold_plain),
        ("fav_taker", fav_taker_h),
        ("fav_maker_bid", OrdersToTrades([o for o in maker_h if o["strategy"] == "fav_maker_bid"], "filled_at")),
        ("fav_maker_mid1", OrdersToTrades([o for o in maker_h if o["strategy"] == "fav_maker_mid1"], "filled_at")),
    ):
        res = Bankroll(trades, hold_days, day_season, tag)
        bank_map[tag] = res
        for kind in ("flat100", "qkelly"):
            bank_rows.extend(res[kind]["rows"])
            bank_metrics.append(res[kind]["metrics"])

    # Coverage summary
    def Quantile(xs, q):
        if not xs:
            return None
        ys = sorted(xs)
        i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
        return ys[i]

    hours = [c["place_to_t60_s"] / 3600.0 for c in coverage]
    cov_summary = {
        "n_priced": len(games),
        "n_tune": len(tune_games),
        "n_holdout": len(hold_games),
        "n_with_placement": len(coverage),
        "maker_skipped": maker_skipped,
        "place_to_t60_hours_min": min(hours) if hours else None,
        "place_to_t60_hours_median": Quantile(hours, 0.5),
        "place_to_t60_hours_max": max(hours) if hours else None,
        "share_le_2h": (sum(1 for h in hours if h <= 2.0) / len(hours)) if hours else None,
        "median_later_candles": Quantile([c["n_later"] for c in coverage], 0.5),
        "median_post_t60_pre_tip": Quantile([c["n_post_t60_pre_tip"] for c in coverage], 0.5),
        "n_zero_post_t60": sum(1 for c in coverage if c["n_post_t60_pre_tip"] == 0),
    }

    # Write CSVs (Basebot column sets).
    WriteCsv(OUT_DIR / f"{BASE}_flb_bins.csv", bin_rows, FLB_BINS_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_flb_grid.csv", GridCsv(grid_main_tune, "tune") + GridCsv(grid_main_h, "holdout"), FLB_GRID_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_flb_grid_alt.csv", GridCsv(grid_alt_tune, "tune") + GridCsv(grid_alt_h, "holdout"), FLB_GRID_FIELDS)
    (OUT_DIR / f"{BASE}_flb_grid_alt.csv").write_text(
        "# Alternate FLB family (same columns as flb_grid.csv). "
        f"This file is the '{alt_op}' grid; flb_grid.csv is the frozen '{main_op}' family.\n"
        + (OUT_DIR / f"{BASE}_flb_grid_alt.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    WriteCsv(OUT_DIR / f"{BASE}_flb_rates_frozen.csv", rate_rows, FLB_RATES_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_maker_adverse.csv", adverse, ADVERSE_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_strategies.csv", strat_rows, STRAT_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_orders.csv", maker_orders, ORDER_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_bankroll.csv", bank_rows, BANKROLL_FIELDS)
    WriteCsv(OUT_DIR / f"{BASE}_bankroll_metrics.csv", bank_metrics, METRICS_FIELDS)

    hold_strats = [r for r in strat_rows if r["split"] == "holdout"]
    tune_strats = [r for r in strat_rows if r["split"] == "tune"]
    any_clear = any(r["clears_3c"] for r in hold_strats)
    verdict = (
        "FAIL (a pre-registered headline strategy clears 3c with a 95% day-block CI above 0 on the holdout)"
        if any_clear else
        "PASS (nothing clears 3c with a 95% day-block CI above 0 on the holdout)"
    )

    for r in hold_strats + tune_strats:
        AppendTrial(trials_path, {
            "phase": "score",
            "split": r["split"],
            "strategy": r["strategy"],
            "n": r["n"],
            "mean_edge": r["mean_edge"],
            "edge_ci95_lo": r["edge_ci95_lo"],
            "edge_ci95_hi": r["edge_ci95_hi"],
            "roi": r["roi"],
            "clears_3c": r["clears_3c"],
            "frozen_hash": frozen_hash,
        })

    # Markdown report
    bin_headers = [
        "ask bin", "N sides", "mean ask", "fair value (win rate)",
        "mean edge/contract at ask+0.02", "95% day-block CI", "ROI per $",
        "edge with Kalshi taker fee",
    ]
    strat_headers = [
        "strategy", "N", "fill rate", "mean price", "fair value", "mean pred. edge",
        "mean realised edge", "95% day-block CI", "ROI per $ [95% CI]",
        "edge w/ Kalshi fee [95% CI]", "clears 3c?",
    ]
    grid_headers = ["x", "tune N", "tune mean edge", "tune ROI", "eligible"]

    def GridMd(rows):
        out = []
        for r in rows:
            out.append([
                f"{r['x']:.2f}",
                str(r["n"]),
                FmtSigned(r["mean_edge"]) if r["n"] else "-",
                FmtSigned(r["roi"]) if r["n"] else "-",
                "True" if r["eligible"] else "False",
            ])
        return out

    frozen_line = "Frozen rule: no eligible grid point; FLB strategies bet N=0."
    if frozen["op"] is not None:
        op_words = "ask >= {0:.2f}".format(frozen["x"]) if frozen["op"] == "geq" else "ask <= {0:.2f}".format(frozen["x"])
        frozen_line = (
            "**Frozen rule: buy the {0} side if {1}.** Tune N={2}, mean edge {3}, ROI {4}. "
            "Frozen tune bin win rates (bins with >= {5} sides) are in `{6}_flb_rates_frozen.csv`. "
            "`flb_rule_3c` also requires frozen bin rate - (ask + 0.02) >= 0.03; "
            "`flb_rule_plain` bets every selected side that matches the frozen threshold."
        ).format(
            "higher-ask" if frozen["op"] == "geq" else "lower-ask",
            op_words, frozen["tune_n"], FmtSigned(frozen["tune_mean_edge"]),
            FmtSigned(frozen["tune_roi"]), MIN_N_FROZEN_BIN, BASE,
        )

    bank_lines = []
    for tag in ("flb_rule_3c", "flb_rule_plain", "fav_taker", "fav_maker_bid", "fav_maker_mid1"):
        res = bank_map[tag]
        for kind in ("flat100", "qkelly"):
            m = res[kind]["metrics"]
            bank_lines.append([
                f"{kind}:{tag}",
                str(m["bets"]),
                FairDollars(m["end_equity"]),
                FmtSigned(m["roi"]),
                FmtCI(m["roi_ci90_day_lo"], m["roi_ci90_day_hi"]),
                FmtCI(m["roi_ci90_widest_lo"], m["roi_ci90_widest_hi"]),
                "-" if not m["ruin_x"] else str(m["ruin_x"]),
            ])

    adv_h = [a for a in adverse if a["season"] == "holdout"]
    adv_lines = []
    for a in adv_h:
        adv_lines.append([
            a["strategy"], a["fill_rule"], "filled" if a["filled"] else "unfilled",
            str(a["n"]),
            Fmt(a["win_rate"]) if a["n"] else "-",
            FairDollars(a["mean_price"]) if a["n"] else "-",
            FmtSigned(a["mean_edge_if_filled_at_limit"]) if a["n"] else "-",
            FmtCI(a["edge_ci95_lo"], a["edge_ci95_hi"]) if a["n"] else "-",
            FmtSigned(a["roi_if_filled_at_limit"]) if a["n"] else "-",
        ])

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    md = f"""# {BASE} - favourite-longshot bias and maker replay (NBA, Kalshi, pre-registered)

Written {now} (ET). Research simulation only; nothing was bet. Pre-registration: `{BASE}_preregistration.md` (frozen before any result). Frozen spec sha256 `{frozen_hash}`.

**Verdict: {verdict}.**

## Setup
- Game set: the 2026-11-13 calibration / bankroll-10k T-60 set (30-min quote age): {len(games)} priced games. Tune (through {TUNE_END}) N={len(tune_games)} games. Holdout ({HOLDOUT_START} through end of 2025-26 plus 2026-27 to date) N={len(hold_games)} games.
- Cost = price paid + $0.02 (Tim's working estimate, unverified). Kalshi fee column: taker 0.07*p*(1-p), maker 0.0175*p*(1-p) (approximate; maker fees only on some markets). Predicted edge = prob - cost; value at >= 3c. Headline = mean REALISED edge per $1 contract (result - cost) with a 95% day-block bootstrap CI ({BOOT_REPS} reps, seed {BOOT_SEED}). 'Clears 3c' = mean >= 0.03 AND CI lower bound > 0. Fair value = realised win rate in $0.xx.
- Bankroll: $10,000 on the holdout only, one continuous bankroll, flat100 and qkelly (0.25 Kelly at cost, cap 2% of start-of-day equity). qkelly needs prob > cost. The `roi_ci90_*` columns in the bankroll metrics file are 90% intervals (Basebot schema); strategy tables use 95%.
- Tidy CSV columns match Basebot `E:\\AgentDocs\\Basebot\\2026-11-13_flb-maker\\` exactly. `t120_bid` / `t120_ask` in the orders file are the placement quote (earliest valid candle before T-60), not a T-120 quote. `flb_grid.csv` is the frozen family's threshold grid; `flb_grid_alt.csv` has the same columns for the other family (comment header notes which is which).

## (a) Favourite-longshot bias: realised win rate vs T-60 ask (each side of each game)
**Tune** (through {TUNE_END}, 2025-26)

{MdTable(bin_headers, BinMdRows(bins_tune))}

**Holdout pooled** ({HOLDOUT_START} through 2026-27 to date)

{MdTable(bin_headers, BinMdRows(bins_h_pool))}

Rule grids tried on TUNE ONLY (eligible = >= {MIN_N_ELIGIBLE} bets). Buy higher-ask if ask >= x:

{MdTable(grid_headers, GridMd(grid_geq))}

Buy lower-ask if ask <= x:

{MdTable(grid_headers, GridMd(grid_leq))}

{frozen_line}

## (b) Maker replay vs taker
- Orders placed at the earliest valid 1-min candle at or before T-60. {len(maker_orders)} maker orders built; {maker_skipped} games lacked a placement quote or were a mid tie and were skipped for maker strategies. Placement-to-T-60 window: min {Fmt(cov_summary['place_to_t60_hours_min'], 2)} h, median {Fmt(cov_summary['place_to_t60_hours_median'], 2)} h, max {Fmt(cov_summary['place_to_t60_hours_max'], 2)} h; share <= 2h = {Fmt(cov_summary['share_le_2h'], 3)}. Median later pre-tip candles = {cov_summary['median_later_candles']}; games with zero post-T-60 pre-tip candles = {cov_summary['n_zero_post_t60']} of {cov_summary['n_with_placement']}.
- Fill 'at_or_through' = a later 1-min candle fully after placement and at or before scheduled tip has a TRADE low <= limit (home) or trade high >= 1-limit (away, implied). 'strict' = through only. Kalshi candles give 1-min trade range and best bid/ask, but no depth, queue or size: **fills are optimistic** (a print at your price does not mean your order was reached). Away fills are implied from the home YES book (no away candles were pulled) and are also optimistic.
- **Candle-coverage caveat: the pulled 1-min candles cover only about 1-2 hours before T-60, plus any pre-tip bars after T-60. This is a late-pregame window, not the listed market open and not a T-120 rest.**

### Holdout (all strategies)
{MdTable(strat_headers, StratMdRows(hold_strats))}

### Tune / descriptive
{MdTable(strat_headers, StratMdRows(tune_strats))}

### Adverse selection: filled vs unfilled maker orders (hypothetical edge at the limit, holdout)
{MdTable(["strategy", "fill rule", "status", "N", "win rate", "mean limit", "mean edge if filled at limit", "95% day-block CI", "ROI if filled at limit"], adv_lines)}

## $10k bankroll on the holdout (maker = at_or_through fills)
{MdTable(["strategy", "bets", "end equity", "ROI", "90% day-block CI", "90% widest CI", "ruin"], bank_lines)}

## Caveats to caption
- $0.02 fee is Tim's working estimate (unverified), labelled `{FEE_BASIS}`. Kalshi fee columns are an approximate published schedule and are not the headline.
- Only home YES candles were pulled; away ask = 1 - home bid, and away maker fills are implied from the home book.
- The candle pull does not contain the listed market open. First candles typically sit 1-2 hours before T-60. Maker fills can only be observed inside this short window. Fills are optimistic (no size, queue, or partials). If bid/ask history is missing at placement the game is skipped for maker strategies.
- Several correlated strategies are reported; no multiple-comparison correction. Default call is PASS unless a headline strategy's holdout 95% day-block CI lies entirely above 0 and mean realised edge >= 0.03.
- Trial registry: `{BASE}_trials.jsonl`. Frozen spec: `{BASE}_spec.json` / `{BASE}_spec.sha256`.
- Research simulation only; nothing was bet.

Elapsed {time.time() - t0:.1f}s. Repo HEAD at write: {repo_hash}.
"""
    (OUT_DIR / f"{BASE}.md").write_text(md, encoding="utf-8")

    run = {
        "written": now,
        "repo_head": repo_hash,
        "frozen_sha256": frozen_hash,
        "verdict": verdict,
        "any_clears_3c": any_clear,
        "coverage": cov_summary,
        "chosen": frozen,
        "elapsed_s": time.time() - t0,
        "out_dir": str(OUT_DIR),
    }
    (OUT_DIR / f"{BASE}_run.json").write_text(json.dumps(run, indent=2, default=str) + "\n", encoding="utf-8")
    AppendTrial(trials_path, {"phase": "done", "verdict": verdict, "sha256": frozen_hash, "elapsed_s": time.time() - t0})

    # Optional bankroll drafts via BankrollPlot if matplotlib is importable.
    try:
        bp = LoadBankrollPlot()
        # Render is attempted later with the Graphtard venv if this venv has no matplotlib.
        _ = bp
    except Exception:
        pass

    print(json.dumps({
        "ok": True,
        "verdict": verdict,
        "frozen_sha256": frozen_hash,
        "n_tune": len(tune_games),
        "n_holdout": len(hold_games),
        "chosen": frozen,
        "out": str(OUT_DIR),
        "elapsed_s": round(time.time() - t0, 1),
    }, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())