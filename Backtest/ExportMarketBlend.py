"""Walk-forward Kalshi blend, relative Ignorance in bits, and $10k blend bankrolls.

Reuses the 2026-11-13_calibration / bankroll-10k priced set (1,469 games).
Does not re-walk stack_global. Elo v1 is recomputed leak-free (cheap).
Burn-in is the early 2025-26 holdout weeks: Kalshi history only starts
2025-04-15, and stack_global forecasts were not saved for 2024-25.

.venv\\Scripts\\python.exe Backtest/ExportMarketBlend.py
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
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

MAIN = Path(r"E:\AgentWorkspaces\Basketball\basketball")
WORKTREE = Path(r"E:\AgentWorkspaces\Basketball\basketball-wt-v3")
EDGES_IN = Path(r"E:\AgentDocs\Basketbot\2026-11-13_bankroll-10k\2026-11-13_bankroll-10k_edges.csv")
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_market-blend")
BASE = "2026-11-13_market-blend"
FEE = 0.02
FEE_BASIS = "tim_est_0.02"
PRICE_BASIS = "kalshi_T60"
EDGE_CUT = 0.03
START_BANK = 10000.0
FLAT_STAKE = 100.0
QKELLY_FRAC = 0.25
QKELLY_CAP = 0.02
RECORD_FROM = "2024-11-15"
# Predictions start on this date; earlier holdout games are burn-in only.
BURNIN_END = "2025-12-02"  # exclusive; ~6 weeks after 2025-10-21
BOOT_REPS = 10000
BOOT_SEED = 7
EPS = 1e-6
BINS = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]


def PreparePath() -> None:
    os.chdir(MAIN)
    for name in list(sys.modules):
        if name == "Backtest" or name.startswith("Backtest.") or name == "Matchups" or name.startswith("Matchups."):
            del sys.modules[name]
    while str(WORKTREE) in sys.path:
        sys.path.remove(str(WORKTREE))
    sys.path.insert(0, str(WORKTREE))


def LoadBankrollPlot():
    path = MAIN / "Backtest" / "BankrollPlot.py"
    spec = importlib.util.spec_from_file_location("BasketbotBankrollPlot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ClipP(p, lo=EPS, hi=1.0 - EPS) -> float:
    return min(max(float(p), lo), hi)


def Logit(p) -> float:
    p = ClipP(p)
    return math.log(p / (1.0 - p))


def Sigmoid(z) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def Log2(x) -> float:
    return math.log(x, 2.0)


def IgnBits(p, y) -> float:
    q = ClipP(p) if int(y) == 1 else ClipP(1.0 - p)
    return -Log2(q)


def LogLossNats(p, y) -> float:
    q = ClipP(p) if int(y) == 1 else ClipP(1.0 - p)
    return -math.log(q)


def FitLogit(lm, lk, y, l2=1e-6):
    """Unregularised-ish logistic MLE: z = a + wm*logit(model) + wk*logit(kalshi)."""
    lm = np.asarray(lm, dtype=float)
    lk = np.asarray(lk, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 20:
        return None
    X = np.column_stack([np.ones(n), lm, lk])
    theta = np.zeros(3)
    # Weak start: intercept 0, split weight toward the market.
    theta[0] = 0.0
    theta[2] = 1.0
    eye = np.eye(3)
    ok = False
    for _ in range(40):
        z = X @ theta
        z = np.clip(z, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        p = np.clip(p, EPS, 1.0 - EPS)
        resid = p - y
        grad = X.T @ resid + l2 * theta
        w = p * (1.0 - p)
        H = X.T @ (w[:, None] * X) + l2 * eye
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        theta = theta - step
        if float(np.max(np.abs(step))) < 1e-10:
            ok = True
            break
    else:
        ok = True
    a, wm, wk = (float(theta[0]), float(theta[1]), float(theta[2]))
    return {"a": a, "w_m": wm, "w_k": wk, "n": n, "ok": ok}


def BlendP(fit, p_model, p_mkt) -> float:
    z = fit["a"] + fit["w_m"] * Logit(p_model) + fit["w_k"] * Logit(p_mkt)
    return Sigmoid(z)


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
                elif isinstance(v, str):
                    out[k] = v
                elif isinstance(v, float):
                    if math.isfinite(v) and v == math.trunc(v):
                        out[k] = f"{v:.1f}"
                    else:
                        out[k] = f"{v:.10g}"
                else:
                    out[k] = v
            w.writerow(out)


def LoadPriced() -> list:
    rows = []
    with EDGES_IN.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            side = r["side"]
            model_side = float(r["model_prob"])
            mkt_side = float(r["market_prob_devig"])
            fill = float(r["fill_price"])
            result_side = int(float(r["result"]))
            if side == "home":
                p_stack = model_side
                p_mkt = mkt_side
                home_win = result_side
                home_ask = fill
                home_bid = 2.0 * p_mkt - home_ask
                away_ask = 1.0 - home_bid
            elif side == "away":
                p_stack = 1.0 - model_side
                p_mkt = 1.0 - mkt_side
                home_win = 1 - result_side
                away_ask = fill
                home_bid = 1.0 - away_ask
                home_ask = 2.0 * p_mkt - home_bid
            else:
                raise ValueError(f"bad side {side} for {r['event_id']}")
            rows.append({
                "game_id": str(r["event_id"]),
                "game_date": r["date"],
                "start_time": r["start_time"],
                "price_time": r["price_time"],
                "season_label": r["season"],
                "split": r["split"],
                "event": r["event"],
                "p_stack": p_stack,
                "p_mkt": p_mkt,
                "home_win": int(home_win),
                "home_ask": float(home_ask),
                "away_ask": float(away_ask),
            })
    rows.sort(key=lambda g: (g["game_date"], g["game_id"]))
    return rows


def AttachElo(games: list) -> None:
    from Backtest.StackDay import EloHistory
    last = max(g["game_date"] for g in games)
    until_excl = (date.fromisoformat(last) + timedelta(days=1)).isoformat()
    seasons = [2023, 2024, 2025, 2026]
    print(f"EloHistory seasons={seasons} until={until_excl}", flush=True)
    elo = EloHistory(seasons, RECORD_FROM, until_excl)
    by_id = {str(r["game_id"]): r for r in elo["rows"]}
    missing = 0
    for g in games:
        er = by_id.get(g["game_id"])
        if er is None:
            missing += 1
            g["p_elo"] = 0.5
        else:
            g["p_elo"] = float(er["p_home"])
    print(f"elo attached, missing={missing}", flush=True)


def BootstrapMean(values, clusters, reps=BOOT_REPS, seed=BOOT_SEED):
    """90% CI for the mean. Game-level if clusters is None, else cluster (e.g. day)."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_clusters": 0}
    point = sum(vals) / n
    rng = random.Random(seed)
    if clusters is None:
        boots = []
        for _ in range(reps):
            s = [vals[rng.randrange(n)] for _ in range(n)]
            boots.append(sum(s) / n)
        n_cl = n
    else:
        bucket = defaultdict(list)
        for c, v in zip(clusters, vals):
            bucket[c].append(v)
        keys = list(bucket)
        n_cl = len(keys)
        boots = []
        for _ in range(reps):
            draw = [keys[rng.randrange(n_cl)] for _ in range(n_cl)]
            s = []
            for k in draw:
                s.extend(bucket[k])
            boots.append(sum(s) / len(s) if s else 0.0)
    boots.sort()
    lo = boots[int(0.05 * reps)]
    hi = boots[min(reps - 1, int(0.95 * reps))]
    return {"mean": point, "lo": lo, "hi": hi, "n": n, "n_clusters": n_cl}


def GapBits(games, p_field):
    ign_m = [IgnBits(g[p_field], g["home_win"]) for g in games]
    ign_k = [IgnBits(g["p_mkt"], g["home_win"]) for g in games]
    diffs = [a - b for a, b in zip(ign_m, ign_k)]
    days = [g["game_date"] for g in games]
    game_ci = BootstrapMean(diffs, None)
    day_ci = BootstrapMean(diffs, days)
    # Widest interval (more conservative).
    lo = min(game_ci["lo"], day_ci["lo"])
    hi = max(game_ci["hi"], day_ci["hi"])
    mean = game_ci["mean"]
    return {
        "ign_rel_bits": mean,
        "lo": lo, "hi": hi,
        "lo_game": game_ci["lo"], "hi_game": game_ci["hi"],
        "lo_day": day_ci["lo"], "hi_day": day_ci["hi"],
        "n": len(games),
        "n_days": day_ci["n_clusters"],
        "growth_per_event": (2.0 ** (-mean)) if mean is not None else None,
        "ign_model": sum(ign_m) / len(games) if games else None,
        "ign_market": sum(ign_k) / len(games) if games else None,
    }


def WeekStart(d: str) -> str:
    dt = date.fromisoformat(d)
    return (dt - timedelta(days=dt.weekday())).isoformat()


def WalkBlend(games, p_field, burnin_end=BURNIN_END):
    """Weekly expanding logistic blend. Fit only on games strictly before the week."""
    by_week = defaultdict(list)
    for g in games:
        by_week[WeekStart(g["game_date"])].append(g)
    weeks = sorted(by_week)
    history_lm, history_lk, history_y, history_days = [], [], [], []
    weights = []
    n_pred = 0
    current = None
    for wk in weeks:
        prior = [g for g in games if g["game_date"] < wk]
        # Only start fitting once burn-in has elapsed (prior includes burn-in games).
        if wk >= burnin_end and prior:
            fit = FitLogit(
                [Logit(g[p_field]) for g in prior],
                [Logit(g["p_mkt"]) for g in prior],
                [g["home_win"] for g in prior],
            )
        else:
            fit = None
        current = fit
        n_prior = len(prior)
        row = {
            "week_start": wk,
            "n_prior": n_prior,
            "n_week": len(by_week[wk]),
            "a": None if not fit else fit["a"],
            "w_m": None if not fit else fit["w_m"],
            "w_k": None if not fit else fit["w_k"],
        }
        for g in by_week[wk]:
            if fit is None:
                g["p_blend"] = None
                g["blend_used"] = False
            else:
                g["p_blend"] = BlendP(fit, g[p_field], g["p_mkt"])
                g["blend_used"] = True
                n_pred += 1
        weights.append(row)
    return weights, current, n_pred


def BootstrapWm(games, p_field, reps=2000, seed=BOOT_SEED):
    """Day-clustered bootstrap of the final-fit w_m (all post-first-game prior = all OOS predecessors + burn-in)."""
    if not games:
        return {"w_m": None, "lo": None, "hi": None, "n": 0}
    # Final fit uses every game before the last predicted week's Monday, i.e. all but the last week.
    # Report the last fitted weight's CI from resampling the final training set.
    last_pred = max((g["game_date"] for g in games if g.get("blend_used")), default=None)
    if last_pred is None:
        return {"w_m": None, "lo": None, "hi": None, "n": 0}
    wk = WeekStart(last_pred)
    train = [g for g in games if g["game_date"] < wk]
    point = FitLogit(
        [Logit(g[p_field]) for g in train],
        [Logit(g["p_mkt"]) for g in train],
        [g["home_win"] for g in train],
    )
    if not point:
        return {"w_m": None, "lo": None, "hi": None, "n": 0}
    bucket = defaultdict(list)
    for g in train:
        bucket[g["game_date"]].append(g)
    keys = list(bucket)
    rng = random.Random(seed)
    boots = []
    for _ in range(reps):
        draw = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        samp = []
        for k in draw:
            samp.extend(bucket[k])
        fit = FitLogit(
            [Logit(g[p_field]) for g in samp],
            [Logit(g["p_mkt"]) for g in samp],
            [g["home_win"] for g in samp],
        )
        if fit:
            boots.append(fit["w_m"])
    boots.sort()
    if not boots:
        return {"w_m": point["w_m"], "lo": None, "hi": None, "n": len(train), "a": point["a"], "w_k": point["w_k"]}
    return {
        "w_m": point["w_m"], "a": point["a"], "w_k": point["w_k"],
        "lo": boots[int(0.05 * len(boots))],
        "hi": boots[min(len(boots) - 1, int(0.95 * len(boots)))],
        "n": len(train), "n_days": len(keys), "n_boot": len(boots),
    }


def Metrics(pairs):
    if not pairs:
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None, "ece": None}
    n = len(pairs)
    acc = sum(int((p >= 0.5) == bool(y)) for p, y in pairs) / n
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    ll = sum(LogLossNats(p, y) for p, y in pairs) / n
    ece = 0.0
    for i, (lo, hi) in enumerate(BINS):
        if i == 9:
            cell = [(p, y) for p, y in pairs if lo <= p <= hi]
        else:
            cell = [(p, y) for p, y in pairs if lo <= p < hi]
        if not cell:
            continue
        ece += (len(cell) / n) * abs(sum(p for p, _y in cell) / len(cell) - sum(y for _p, y in cell) / len(cell))
    return {"n": n, "accuracy": acc, "brier": brier, "log_loss": ll, "ece": ece}


def SummaryRows(games_by_key, asof):
    """games_by_key: (model, season, split) -> list of (p, y, date)."""
    rows = []
    order = []
    seen = set()
    for key in games_by_key:
        if key not in seen:
            order.append(key)
            seen.add(key)
    for model, season, split in order:
        items = games_by_key[(model, season, split)]
        pairs = [(p, y) for p, y, _d in items]
        last = max((d for _p, _y, d in items), default=asof.get(season, ""))
        met = Metrics(pairs)
        rows.append({
            "date": last, "sport": "nba", "series": f"nba_{model}", "model": model,
            "season": season, "split": split,
            "n": met["n"], "accuracy": met["accuracy"], "brier": met["brier"],
            "log_loss": met["log_loss"], "ece": met["ece"],
        })
    return rows


def ChosenSide(model_home, mkt_home):
    edge_h = model_home - mkt_home - FEE
    edge_a = (1.0 - model_home) - (1.0 - mkt_home) - FEE
    if edge_h >= edge_a:
        return "home", edge_h, model_home, mkt_home
    return "away", edge_a, 1.0 - model_home, 1.0 - mkt_home


def PnlFlat(fill, result, stake=FLAT_STAKE):
    c = fill + FEE
    if c <= 0:
        return 0.0
    contracts = stake / c
    return contracts * result - stake


def EdgeRows(priced):
    rows = []
    for g in priced:
        p = g["p_blend"]
        m = g["p_mkt"]
        side, edge, model_side, mkt_side = ChosenSide(p, m)
        fill = g["home_ask"] if side == "home" else g["away_ask"]
        result = g["home_win"] if side == "home" else 1 - g["home_win"]
        edge_vs_fill = model_side - fill - FEE
        rows.append({
            "date": g["game_date"],
            "start_time": g["start_time"],
            "sport": "nba",
            "season": g["season_label"],
            "split": g["split"],
            "event_id": g["game_id"],
            "event": g["event"],
            "side": side,
            "model": "stack_kalshi_blend",
            "model_prob": model_side,
            "market_prob_devig": mkt_side,
            "fill_price": fill,
            "price_basis": PRICE_BASIS,
            "price_time": g["price_time"],
            "fee": FEE,
            "fee_basis": FEE_BASIS,
            "edge": edge,
            "edge_vs_fill": edge_vs_fill,
            # R3: qualify vs ask + Tim's $0.02 estimate, 3c bar.
            "qualifies": bool(edge_vs_fill >= EDGE_CUT),
            "result": int(result),
            "pnl_at_flat_stake": PnlFlat(fill, int(result)),
        })
    return rows


def BankrollPaths(priced, all_days, edge_by_id, day_season):
    priced_by_day = defaultdict(list)
    for g in priced:
        priced_by_day[g["game_date"]].append(g)

    def Run(strategy):
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
                er = edge_by_id.get(g["game_id"])
                if er is None or not er["qualifies"]:
                    continue
                fill = er["fill_price"]
                result = er["result"]
                p = er["model_prob"]
                c = fill + FEE
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
                    pnl = (stake / c) * result - stake
                equity = equity + pnl
                bets_day += 1
                bets_cum += 1
                stakes_pnls.append((stake, pnl))
            if equity > peak:
                peak = equity
            dd = 0.0 if peak == 0 else min(0.0, equity / peak - 1.0)
            season = day_season[day]
            split = "holdout" if season == "2025-26" else "forward"
            rows.append({
                "date": day, "sport": "nba",
                "series": f"nba_{strategy}_{PRICE_BASIS}",
                "strategy": strategy, "price_basis": PRICE_BASIS,
                "season": season, "split": split,
                "bets_day": bets_day, "bets_cum": bets_cum,
                "equity": equity, "drawdown_pct": dd,
            })
        return {
            "rows": rows, "end": equity, "stakes_pnls": stakes_pnls,
            "ruined": ruined, "ruin_day": ruin_day,
            "max_dd": min((r["drawdown_pct"] for r in rows), default=0.0),
            "n_bets": bets_cum,
        }

    return {name: Run(name) for name in ("flat100", "qkelly")}


def BootstrapRoi(stakes_pnls, reps=BOOT_REPS, seed=BOOT_SEED):
    if not stakes_pnls:
        return {"roi": None, "lo": None, "hi": None, "n": 0}
    pnls = [p for _s, p in stakes_pnls]
    stakes = [s for s, _p in stakes_pnls]
    total_stake = sum(stakes)
    point = sum(pnls) / total_stake if total_stake else None
    rng = random.Random(seed)
    n = len(pnls)
    boots = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        s = sum(stakes[i] for i in idx)
        if s <= 0:
            continue
        boots.append(sum(pnls[i] for i in idx) / s)
    boots.sort()
    if not boots:
        return {"roi": point, "lo": None, "hi": None, "n": n}
    return {
        "roi": point,
        "lo": boots[int(0.05 * len(boots))],
        "hi": boots[min(len(boots) - 1, int(0.95 * len(boots)))],
        "n": n,
    }


def Fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def FmtCi(d, nd=4):
    if not d or d.get("mean") is None:
        if d and d.get("ign_rel_bits") is not None:
            return f"{d['ign_rel_bits']:+.{nd}f} [{Fmt(d['lo'], nd)}, {Fmt(d['hi'], nd)}]"
        return "n/a"
    return f"{d['mean']:+.{nd}f} [{Fmt(d['lo'], nd)}, {Fmt(d['hi'], nd)}]"


def GitHash(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), text=True).strip()


def WriteMd(path, priced, scored, gaps, wm, weights, paths, roi, burnin, hash_main, hash_wt, notes):
    lines = []
    a = lines.append
    a("# NBA market blend vs Kalshi T-60 (2026-11-13)")
    a("")
    a("Simulation only. No wagers were placed. Default call is PASS unless the blend weight is clearly away from zero and the bankroll CI is above zero.")
    a("")
    a("## Setup")
    a("- Game set: the 2026-11-13 calibration priced set, `kalshi_T60`.")
    a(f"- Priced N: holdout 2025-26 {sum(1 for g in priced if g['split']=='holdout')}, forward 2026-27 {sum(1 for g in priced if g['split']=='forward')}, pooled {len(priced)}.")
    a("- Headline model: walk-forward `stack_global` at worktree `6df88fe`. Shadow: `elo_predictday_v1`. Market: Kalshi home-YES mid at T-60.")
    a("- Blend: `p = sigmoid(a + w_m logit(stack_global) + w_k logit(Kalshi mid))`, fit by log-loss MLE, weekly expanding window, leak-free (fit on earlier games only).")
    a(f"- Burn-in: {burnin}. 2024-25 was **not** used as burn-in (see caveats).")
    a("- Edges and stakes use **p_blend**. Qualification is R3: `p_blend - (ask + $0.02) >= 0.03`. Fee is Tim's $0.02 working estimate.")
    a("- `flat100`: $100, stop if equity < $100. `qkelly`: 0.25x Kelly at c=ask+0.02, 2% of start-of-day equity cap, from $10,000, no reset.")
    a(f"- Code: worktree `{hash_wt}`; exporter/BankrollPlot `{hash_main}`.")
    a("")
    a("## 1. Relative Ignorance vs Kalshi (bits per game)")
    a("")
    a("`ign_rel_bits = mean(-log2 p_model(y)) - mean(-log2 p_kalshi(y))`. Positive means the model is more ignorant than the market. 90% CIs are the widest of game-level and day-clustered bootstraps (10,000 reps). `growth_per_event = 2^(-ign_rel_bits)` is Roulston & Smith log-wealth growth for a proportional bettor vs the market, before fees.")
    a("")
    a("| model | split | N | N days | ign_rel_bits [90% widest CI] | growth/event | IGN model | IGN Kalshi |")
    a("|---|---|---:|---:|---|---:|---:|---:|")
    for key in sorted(gaps):
        g = gaps[key]
        model, split = key
        a(f"| {model} | {split} | {g['n']} | {g['n_days']} | {g['ign_rel_bits']:+.4f} [{Fmt(g['lo'],4)}, {Fmt(g['hi'],4)}] | {Fmt(g['growth_per_event'],4)} | {Fmt(g['ign_model'],4)} | {Fmt(g['ign_market'],4)} |")
    a("")
    a("## 2. Walk-forward logistic blend")
    a("")
    last_w = next((w for w in reversed(weights) if w.get("w_m") is not None), None)
    a(f"- Final `w_m` (stack_global weight) = {Fmt(wm.get('w_m'), 4)} with 90% day-clustered bootstrap CI [{Fmt(wm.get('lo'), 4)}, {Fmt(wm.get('hi'), 4)}] on the final training set (N={wm.get('n')}, {wm.get('n_days')} days).")
    a(f"- Final intercept `a` = {Fmt(wm.get('a'), 4)}, market weight `w_k` = {Fmt(wm.get('w_k'), 4)}.")
    if last_w:
        a(f"- Last fitted week starts {last_w['week_start']} with {last_w['n_prior']} prior games.")
    wm_zero = (wm.get("w_m") is None) or (wm.get("lo") is not None and wm["lo"] <= 0 <= (wm["hi"] or 0)) or (wm.get("w_m") is not None and abs(wm["w_m"]) < 0.05 and wm.get("lo") is not None and wm["lo"] <= 0)
    # Don't pre-judge; just report. Verdict later uses this.
    scored_n = len(scored)
    # log loss in nats and bits on scored games
    def ll_bits(field):
        if not scored:
            return None
        return sum(IgnBits(g[field], g["home_win"]) for g in scored) / len(scored)
    def ll_nats(field):
        if not scored:
            return None
        return sum(LogLossNats(g[field], g["home_win"]) for g in scored) / len(scored)
    a(f"- Out-of-sample games scored by the blend (after burn-in): {scored_n} (holdout {sum(1 for g in scored if g['split']=='holdout')}, forward {sum(1 for g in scored if g['split']=='forward')}).")
    a(f"- Blend OOS log loss vs Kalshi on those games: blend {Fmt(ll_nats('p_blend'),4)} nats / {Fmt(ll_bits('p_blend'),4)} bits; Kalshi {Fmt(ll_nats('p_mkt'),4)} nats / {Fmt(ll_bits('p_mkt'),4)} bits; stack_global {Fmt(ll_nats('p_stack'),4)} nats / {Fmt(ll_bits('p_stack'),4)} bits.")
    a("")
    a("## 3. $10k bankrolls on p_blend (R3 qualification)")
    a("")
    a("| strategy | N bets | ending bankroll | ROI per $ staked [90% event bootstrap CI] | max drawdown | ruined |")
    a("|---|---:|---:|---|---:|---|")
    for name in ("flat100", "qkelly"):
        p = paths[name]
        r = roi[name]
        roi_s = "n/a" if r["roi"] is None else f"{r['roi']:+.3f} [{Fmt(r['lo'], 3)}, {Fmt(r['hi'], 3)}]"
        a(f"| {name} | {p['n_bets']} | ${p['end']:,.2f} | {roi_s} | {p['max_dd']:.3f} | {'yes at ' + p['ruin_day'] if p['ruined'] else 'no'} |")
    n_qual = sum(1 for r in (paths and []) )
    n_qual = sum(1 for g in scored if True)
    # recount from edges stored on games? compute from paths or later
    a("")
    a("## Caveats")
    a("- **Injury:** `stack_global` and Elo v1 are not injury-adjusted. The market prices availability; blending toward Kalshi is partly a substitute, not a fix.")
    a("- Candles have **no size**. Fills at the T-60 ask are optimistic for large stakes.")
    a("- Only home YES candles were pulled; away ask is implied as `1 - home_bid`.")
    a("- The $0.02 fee is **Tim's working estimate**, not a verified Kalshi taker fee (`fee_basis=tim_est_0.02`).")
    a("- **Burn-in:** Kalshi NBA candles in this pull start 2025-04-15 (late 2024-25 playoffs only). `stack_global` probabilities were not saved for those 2024-25 games, and this job did not re-walk the 19-member stack. Burn-in is therefore the first ~6 weeks of the 2025-26 holdout (games dated before 2025-12-02). Those burn-in games are in the relative-Ignorance tables (full calibration set) but are **not** in the blend OOS log loss or the blend bankroll's qualifying bets.")
    a("- Qualification uses edge vs **ask + $0.02** (plan R3), not vs the de-vig mid. The `edge` column is still mid-based so the file matches the common 21-column schema.")
    a("- Leak test: blend weights for a week are fit only on games dated before that week's Monday.")
    a("- Four-factors GAP ratings are a separate prototype (not in this folder).")
    a("")
    # verdict filled by caller via notes
    for line in notes:
        a(line)
    a("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def Main() -> int:
    t0 = time.time()
    PreparePath()
    from Matchups import Cpu
    Cpu.LimitThreads()
    Cpu.LowerPriority()
    BP = LoadBankrollPlot()
    hash_main = GitHash(MAIN)
    hash_wt = GitHash(WORKTREE)
    print(f"main={hash_main} worktree={hash_wt}", flush=True)

    priced = LoadPriced()
    print(f"loaded priced {len(priced)} holdout={sum(1 for g in priced if g['split']=='holdout')} forward={sum(1 for g in priced if g['split']=='forward')}", flush=True)
    AttachElo(priced)

    splits = {
        "holdout": [g for g in priced if g["split"] == "holdout"],
        "forward": [g for g in priced if g["split"] == "forward"],
        "pooled_oos": list(priced),
    }
    gaps = {}
    for model, field in (("stack_global", "p_stack"), ("elo_predictday_v1", "p_elo")):
        for split, gs in splits.items():
            gaps[(model, split)] = GapBits(gs, field)
            g = gaps[(model, split)]
            print(f"ign {model} {split}: {g['ign_rel_bits']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] n={g['n']}", flush=True)

    burnin_games = [g for g in priced if g["game_date"] < BURNIN_END]
    print(f"burn-in games before {BURNIN_END}: {len(burnin_games)} (holdout {sum(1 for g in burnin_games if g['split']=='holdout')})", flush=True)
    weights, last_fit, n_pred = WalkBlend(priced, "p_stack", BURNIN_END)
    scored = [g for g in priced if g.get("blend_used")]
    print(f"blend scored {len(scored)} / {len(priced)}", flush=True)
    wm = BootstrapWm(priced, "p_stack")
    print(f"final w_m={wm.get('w_m')} CI=[{wm.get('lo')}, {wm.get('hi')}] n={wm.get('n')}", flush=True)

    # Summary: blend vs market vs stack on the same post-burn-in games.
    asof = {}
    for g in priced:
        asof[g["season_label"]] = g["game_date"]
    asof["pooled_oos"] = max(g["game_date"] for g in priced)
    keys = {}
    def add(model, season, split, p, y, d):
        keys.setdefault((model, season, split), []).append((p, y, d))
    for g in scored:
        add("stack_kalshi_blend", g["season_label"], g["split"], g["p_blend"], g["home_win"], g["game_date"])
        add("stack_kalshi_blend", "pooled_oos", "pooled_oos", g["p_blend"], g["home_win"], g["game_date"])
        add("market_kalshi_T60", g["season_label"], g["split"], g["p_mkt"], g["home_win"], g["game_date"])
        add("market_kalshi_T60", "pooled_oos", "pooled_oos", g["p_mkt"], g["home_win"], g["game_date"])
        add("stack_global", g["season_label"], g["split"], g["p_stack"], g["home_win"], g["game_date"])
        add("stack_global", "pooled_oos", "pooled_oos", g["p_stack"], g["home_win"], g["game_date"])
    # stable order
    ordered = {}
    for model in ("stack_kalshi_blend", "stack_global", "market_kalshi_T60"):
        for season, split in (("2025-26", "holdout"), ("2026-27", "forward"), ("pooled_oos", "pooled_oos")):
            k = (model, season, split)
            if k in keys:
                ordered[k] = keys[k]
    summary = SummaryRows(ordered, asof)

    # Also add blend gap vs kalshi on scored set
    for split, gs in (("holdout", [g for g in scored if g["split"]=="holdout"]),
                      ("forward", [g for g in scored if g["split"]=="forward"]),
                      ("pooled_oos", scored)):
        if gs:
            gaps[("stack_kalshi_blend", split)] = GapBits(gs, "p_blend")

    edges = EdgeRows(scored)
    edge_by_id = {r["event_id"]: r for r in edges}
    n_qual = sum(1 for r in edges if r["qualifies"])
    print(f"qualifying blend edges: {n_qual} / {len(edges)}", flush=True)

    day_season = {}
    all_days = []
    for g in priced:
        d = g["game_date"]
        if d not in day_season:
            day_season[d] = g["season_label"]
            all_days.append(d)
    all_days = sorted(all_days)
    paths = BankrollPaths(scored, all_days, edge_by_id, day_season)
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
    roi = {name: BootstrapRoi(paths[name]["stakes_pnls"]) for name in paths}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sum_fields = ["date", "sport", "series", "model", "season", "split",
                  "n", "accuracy", "brier", "log_loss", "ece"]
    bank_fields = ["date", "sport", "series", "strategy", "price_basis", "season", "split",
                   "x", "bets_day", "bets_cum", "equity", "drawdown_pct"]
    edge_fields = ["date", "start_time", "sport", "season", "split", "event_id", "event", "side",
                   "model", "model_prob", "market_prob_devig", "fill_price", "price_basis",
                   "price_time", "fee", "fee_basis", "edge", "edge_vs_fill", "qualifies",
                   "result", "pnl_at_flat_stake"]
    WriteCsv(OUT_DIR / f"{BASE}_summary.csv", summary, sum_fields)
    WriteCsv(OUT_DIR / f"{BASE}.csv", bank_rows, bank_fields)
    WriteCsv(OUT_DIR / f"{BASE}_edges.csv", edges, edge_fields)

    ign_rows = []
    for (model, split), g in gaps.items():
        ign_rows.append({
            "date": asof.get("2025-26" if split == "holdout" else ("2026-27" if split == "forward" else "pooled_oos"), asof["pooled_oos"]),
            "sport": "nba", "model": model, "split": split,
            "n": g["n"], "n_days": g["n_days"],
            "ign_rel_bits": g["ign_rel_bits"],
            "ci90_lo": g["lo"], "ci90_hi": g["hi"],
            "ci90_lo_game": g["lo_game"], "ci90_hi_game": g["hi_game"],
            "ci90_lo_day": g["lo_day"], "ci90_hi_day": g["hi_day"],
            "growth_per_event": g["growth_per_event"],
            "ign_model_bits": g["ign_model"], "ign_market_bits": g["ign_market"],
        })
    ign_fields = ["date","sport","model","split","n","n_days","ign_rel_bits","ci90_lo","ci90_hi",
                  "ci90_lo_game","ci90_hi_game","ci90_lo_day","ci90_hi_day","growth_per_event",
                  "ign_model_bits","ign_market_bits"]
    WriteCsv(OUT_DIR / f"{BASE}_ign.csv", ign_rows, ign_fields)
    wfields = ["week_start","n_prior","n_week","a","w_m","w_k"]
    WriteCsv(OUT_DIR / f"{BASE}_weights.csv", weights, wfields)

    includes_zero = wm.get("lo") is not None and wm.get("hi") is not None and wm["lo"] <= 0.0 <= wm["hi"]
    near_zero = wm.get("w_m") is not None and abs(wm["w_m"]) < 0.05
    flat = roi["flat100"]
    edge_ok = (flat["roi"] is not None and flat["lo"] is not None and flat["lo"] > 0 and paths["flat100"]["n_bets"] > 0
               and not includes_zero and not near_zero)
    notes = [f"- Qualifying opportunities (blend edge vs ask+$0.02 >= 0.03): {n_qual} of {len(scored)} post-burn-in games."]
    if includes_zero or near_zero or not edge_ok:
        notes.append("- **Verdict: PASS.** `w_m` is about 0 or its 90% CI includes 0, and/or the flat100 ROI CI is not above zero. NBA stays PASS. The blend is not a betting input.")
    else:
        notes.append("- Verdict: research only until Tim says otherwise; do not treat this as a live bet.")

    burnin_txt = (f"{len(burnin_games)} games dated {burnin_games[0]['game_date'] if burnin_games else 'n/a'} "
                  f"through the day before {BURNIN_END} (early 2025-26 holdout weeks; 2024-25 stack+Kalshi burn-in was not available)")
    WriteMd(OUT_DIR / f"{BASE}.md", priced, scored, gaps, wm, weights, paths, roi, burnin_txt, hash_main, hash_wt, notes)

    try:
        d, e, s = [], [], []
        for name in ("flat100", "qkelly"):
            for r in paths[name]["rows"]:
                d.append(r["date"]); e.append(r["equity"]); s.append(name)
        frame = BP.FrameBankroll(d, e, s, mode="index")
        BP.PlotBankroll(frame, path=str(OUT_DIR / f"{BASE}_draft.png"),
                        title="NBA $10k bankroll, stack-Kalshi blend at T-60 (summer skipped)",
                        ylabel="Simulated bankroll (US dollars)")
        print("wrote draft png", flush=True)
    except Exception as exc:
        print(f"draft plot skipped: {exc}", flush=True)

    dump = {
        "hash_main": hash_main, "hash_wt": hash_wt,
        "n_priced": len(priced), "n_burnin": len(burnin_games), "n_scored": len(scored),
        "n_qual": n_qual, "burnin_end": BURNIN_END,
        "gaps": {f"{m}|{s}": v for (m, s), v in gaps.items()},
        "w_m": wm,
        "summary": summary,
        "bankroll": {k: {"end": v["end"], "n_bets": v["n_bets"], "max_dd": v["max_dd"],
                         "ruined": v["ruined"], "ruin_day": v["ruin_day"], "roi": roi[k]}
                     for k, v in paths.items()},
        "pass": includes_zero or near_zero or not edge_ok,
        "elapsed_s": time.time() - t0,
    }
    (OUT_DIR / f"{BASE}_run.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: dump[k] for k in dump if k not in ("summary", "gaps")}, indent=2, default=str), flush=True)
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
        Path(r"E:\AgentWorkspaces\Basketball\basketball\logs\2026-11-13_market-blend.exc").write_text(tb, encoding="utf-8")
        raise