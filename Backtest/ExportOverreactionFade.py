"""Pre-registered Kalshi overreaction fade and COD underperformer variant.

Split: fit cutoffs only on games through 2026-01-31, write the frozen spec and
its hash, then score the holdout (2026-02-01 through the end of 2025-26, plus
2026-27 to date). The 1,469-game kalshi_T60 priced set is reused.

IMPORTANT: the candle pull does not contain the listed market open. The first
available candle is typically 1-2 hours before T-60 and 1-2+ days after
markets.json open_time. The fade is therefore a late-pregame drift fade.
That is a data limitation, not a silent assumption.

.venv\\Scripts\\python.exe Backtest/ExportOverreactionFade.py
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

MAIN = Path(r"E:\Basketball\basketball")
EDGES_IN = Path(r"E:\AgentDocs\Basketbot\2026-11-13_bankroll-10k\2026-11-13_bankroll-10k_edges.csv")
KALSHI_MARKETS = MAIN / "data" / "inplay" / "kalshi" / "markets.json"
KALSHI_CANDLES = MAIN / "data" / "inplay" / "kalshi" / "candles"
GAMES_META = MAIN / "data" / "inplay" / "games.pkl"
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_overreaction-fade")
BASE = "2026-11-13_overreaction-fade"
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
TUNE_END = "2026-01-31"          # inclusive
HOLDOUT_START = "2026-02-01"
COD_MIN_PRIOR = 6
BOOT_REPS = 10000
BOOT_SEED = 7
ET = ZoneInfo("America/New_York")

# Pre-registered trial names (declared before any holdout look).
# Cutoff values are fitted on tune only, then frozen.
PRIMARY = "fade_top_decile"
TRIALS_PRE_REG = (
    "fade_top_decile",           # headline: |move| in tune decile 10
    "fade_top_two_deciles",      # robustness: deciles 9-10
    "fade_all_nonzero",          # baseline: any nonzero window-open to T-60 move
    "fade_positive_tune_deciles",  # fitted: tune deciles with mean P/L after fee > 0
    "fade_top_decile_value3c",   # overlay: top decile AND p_tune - (ask+0.02) >= 0.03
    "cod_bottom_decile",         # variant: back the underperformer, bottom tune decile
    "cod_all_min6",              # COD baseline: any game with both teams >= 6 priors
)


def LimitCpu() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def LoadBankrollPlot():
    path = MAIN / "Backtest" / "BankrollPlot.py"
    spec = importlib.util.spec_from_file_location("BasketbotBankrollPlot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def Mid(bid, ask):
    if bid is None or ask is None:
        return None
    return (float(bid) + float(ask)) / 2.0


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


def GitHash(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), text=True).strip()


def CanonicalHash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _json_default(o):
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def ValidCandle(c) -> bool:
    if c.get("t") is None or c.get("bid") is None or c.get("ask") is None:
        return False
    bid, ask = float(c["bid"]), float(c["ask"])
    if ask < bid or bid < 0 or ask > 1 or bid > 1 or ask <= 0:
        return False
    mid = Mid(bid, ask)
    return mid is not None and 0.0 < mid < 1.0


def QuoteFromCandle(c) -> dict:
    bid, ask = float(c["bid"]), float(c["ask"])
    return {"bid": bid, "ask": ask, "mid": Mid(bid, ask), "t": int(c["t"])}


def Teams(event: str):
    away, home = event.split(" @ ")
    return home.strip(), away.strip()


def SplitOf(date: str) -> str:
    return "tune" if date <= TUNE_END else "holdout"


def SeasonOf(date: str) -> str:
    return "2025-26" if date <= "2026-06-30" else "2026-27"


def PercentileCuts(values, n_bins=10):
    xs = sorted(float(v) for v in values)
    if not xs:
        return []
    cuts = []
    for i in range(1, n_bins):
        # inclusive rank
        pos = (i / n_bins) * (len(xs) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            cuts.append(xs[lo])
        else:
            w = pos - lo
            cuts.append(xs[lo] * (1 - w) + xs[hi] * w)
    return cuts


def AssignBin(value, cuts) -> int:
    for i, c in enumerate(cuts, start=1):
        if value <= c:
            return i
    return len(cuts) + 1


def PnlContract(ask, won) -> float:
    return (1.0 if int(won) else 0.0) - (float(ask) + FEE)


def BootstrapDayMean(pairs, reps=BOOT_REPS, seed=BOOT_SEED, level=0.95):
    """pairs: (day, value). 95% (or level) day-block bootstrap CI for the mean of values."""
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
    return {"mean": point, "lo": lo, "hi": hi, "n": len(values), "n_days": n_cl, "level": level}


def LoadSchedAndMarkets():
    import pandas as pd
    frame = pickle.loads(GAMES_META.read_bytes())
    sched = {}
    for rec in frame.itertuples(index=False):
        key = (str(rec.date_et)[:10], rec.home, rec.away)
        st = None if pd.isna(rec.sched_ts) else int(rec.sched_ts)
        tk = None if pd.isna(rec.kalshi_ticker) else str(rec.kalshi_ticker)
        sched[key] = {"sched_ts": st, "ticker": tk}
    markets = json.loads(KALSHI_MARKETS.read_text(encoding="utf-8"))
    open_by_ticker, ticker_by_key = {}, {}
    for m in markets:
        if not m.get("is_home"):
            continue
        key = (m["date_et"], m["home"], m["away"])
        ticker_by_key[key] = m["ticker"]
        open_by_ticker[m["ticker"]] = m.get("open_time")
    return sched, open_by_ticker, ticker_by_key


def LoadPricedEdges():
    rows = []
    with EDGES_IN.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            home, away = Teams(r["event"])
            side = r["side"]
            mkt_side = float(r["market_prob_devig"])
            fill = float(r["fill_price"])
            res_side = int(float(r["result"]))
            if side == "home":
                p_mkt = mkt_side
                home_win = res_side
                home_ask = fill
                home_bid = 2.0 * p_mkt - home_ask
                away_ask = 1.0 - home_bid
            else:
                p_mkt = 1.0 - mkt_side
                home_win = 1 - res_side
                away_ask = fill
                home_bid = 1.0 - away_ask
                home_ask = 2.0 * p_mkt - home_bid
            rows.append({
                "game_id": str(r["event_id"]),
                "game_date": r["date"],
                "start_time": r["start_time"],
                "price_time": r["price_time"],
                "event": r["event"],
                "home": home, "away": away,
                "p_mkt": p_mkt,
                "home_win": int(home_win),
                "home_ask_t60_edge": float(home_ask),
                "away_ask_t60_edge": float(away_ask),
                "season_label": r["season"],
                "orig_split": r["split"],
                "eval_split": SplitOf(r["date"]),
            })
    rows.sort(key=lambda g: (g["game_date"], g["game_id"]))
    return rows


def AttachOpens(games, sched, open_by_ticker, ticker_by_key):
    dropped = []
    out = []
    for g in games:
        key = (g["game_date"], g["home"], g["away"])
        sch = sched.get(key, {})
        ticker = sch.get("ticker") or ticker_by_key.get(key)
        sched_ts = sch.get("sched_ts")
        g = dict(g)
        g["ticker"] = ticker
        g["sched_ts"] = sched_ts
        g["listed_open_time"] = open_by_ticker.get(ticker) if ticker else None
        # Edge-file T-60 fills are the fallback when a window-open quote is missing (COD variant).
        g["home_ask"] = g["home_ask_t60_edge"]
        g["away_ask"] = g["away_ask_t60_edge"]
        g["home_mid"] = g["p_mkt"]
        g["away_mid"] = 1.0 - g["p_mkt"]
        if not ticker or not sched_ts:
            dropped.append({"game_id": g["game_id"], "date": g["game_date"], "reason": "no_ticker_or_sched"})
            g["has_open"] = False
            g["drop_reason"] = "no_ticker_or_sched"
            out.append(g)
            continue
        path = KALSHI_CANDLES / f"{ticker}.json"
        if not path.exists():
            dropped.append({"game_id": g["game_id"], "date": g["game_date"], "reason": "no_candle_file"})
            g["has_open"] = False
            g["drop_reason"] = "no_candle_file"
            out.append(g)
            continue
        candles = [c for c in (json.loads(path.read_text(encoding="utf-8")).get("candles") or []) if ValidCandle(c)]
        if not candles:
            dropped.append({"game_id": g["game_id"], "date": g["game_date"], "reason": "no_valid_candles"})
            g["has_open"] = False
            g["drop_reason"] = "no_valid_candles"
            out.append(g)
            continue
        target = int(sched_ts) - T60_SEC
        before_t60 = [c for c in candles if int(c["t"]) <= target]
        if not before_t60:
            dropped.append({"game_id": g["game_id"], "date": g["game_date"], "reason": "no_quote_at_or_before_T-60"})
            g["has_open"] = False
            g["drop_reason"] = "no_quote_at_or_before_T-60"
            out.append(g)
            continue
        t60c = before_t60[-1]
        age = target - int(t60c["t"])
        if age > STALE_SEC:
            dropped.append({"game_id": g["game_id"], "date": g["game_date"], "reason": "stale_t60_>30min"})
            g["has_open"] = False
            g["drop_reason"] = "stale_t60_>30min"
            out.append(g)
            continue
        t60 = QuoteFromCandle(t60c)
        t60["age_s"] = age
        # Window-open: first valid candle at or before T-60 (first in the pulled window).
        first = before_t60[0]
        opn = QuoteFromCandle(first)
        if int(opn["t"]) == int(t60["t"]):
            dropped.append({"game_id": g["game_id"], "date": g["game_date"], "reason": "open_equals_t60"})
            g["has_open"] = False
            g["drop_reason"] = "open_equals_t60"
            out.append(g)
            continue
        listed = g["listed_open_time"]
        listed_ts = None
        if listed:
            listed_ts = int(datetime.fromisoformat(listed.replace("Z", "+00:00")).timestamp())
        g["has_open"] = True
        g["drop_reason"] = None
        g["open"] = opn
        g["t60"] = t60
        g["open_to_t60_s"] = int(t60["t"]) - int(opn["t"])
        g["listed_open_to_first_s"] = (int(opn["t"]) - listed_ts) if listed_ts is not None else None
        g["true_open_missing"] = (listed_ts is None) or (int(opn["t"]) - listed_ts > 3600)
        # Last pregame mid: last valid candle at or before scheduled tip.
        pre = [c for c in candles if int(c["t"]) <= int(sched_ts)]
        g["last_pre"] = QuoteFromCandle(pre[-1]) if pre else None
        # Recompute fills from this T-60 quote (source of truth for the fade).
        g["home_bid"] = t60["bid"]
        g["home_ask"] = t60["ask"]
        g["home_mid"] = t60["mid"]
        g["away_ask"] = 1.0 - t60["bid"]
        g["away_mid"] = 1.0 - t60["mid"]
        g["open_home_mid"] = opn["mid"]
        g["dmid_home"] = t60["mid"] - opn["mid"]
        g["abs_move"] = abs(g["dmid_home"])
        if g["dmid_home"] > 0:
            g["rise_side"] = "home"
            g["fade_side"] = "away"
        elif g["dmid_home"] < 0:
            g["rise_side"] = "away"
            g["fade_side"] = "home"
        else:
            g["rise_side"] = None
            g["fade_side"] = None
        out.append(g)
    return out, dropped


def AttachCOD(games):
    """Leak-free season residual: sum(result - kalshi_mid) over prior priced games."""
    stats = defaultdict(lambda: {"n": 0, "exp": 0.0, "wins": 0.0})  # key (season, team)
    for g in games:
        season = g["season_label"]
        hs = stats[(season, g["home"])]
        aws = stats[(season, g["away"])]
        g["home_prior_n"] = hs["n"]
        g["away_prior_n"] = aws["n"]
        g["home_cod"] = (hs["wins"] - hs["exp"]) if hs["n"] else 0.0
        g["away_cod"] = (aws["wins"] - aws["exp"]) if aws["n"] else 0.0
        g["cod_ready"] = hs["n"] >= COD_MIN_PRIOR and aws["n"] >= COD_MIN_PRIOR
        if g["cod_ready"]:
            if g["home_cod"] < g["away_cod"]:
                g["cod_side"] = "home"
            elif g["away_cod"] < g["home_cod"]:
                g["cod_side"] = "away"
            else:
                g["cod_side"] = None
            g["cod_under"] = min(g["home_cod"], g["away_cod"])
        else:
            g["cod_side"] = None
            g["cod_under"] = None
        # Update after the game. Use T-60 mid as the Kalshi-implied expectation.
        # Even games without a window-open still have a T-60 mid from the edges file.
        p_home = g["p_mkt"]
        hs["n"] += 1
        aws["n"] += 1
        hs["exp"] += p_home
        aws["exp"] += 1.0 - p_home
        hs["wins"] += g["home_win"]
        aws["wins"] += 1 - g["home_win"]
    return games


def SideAsk(g, side) -> float:
    if side == "home":
        return g.get("home_ask", g.get("home_ask_t60_edge"))
    return g.get("away_ask", g.get("away_ask_t60_edge"))


def SideWon(g, side) -> int:
    return int(g["home_win"] if side == "home" else 1 - g["home_win"])


def SideLastMid(g, side):
    lp = g.get("last_pre")
    if not lp:
        return None
    mid_h = lp["mid"]
    return mid_h if side == "home" else (1.0 - mid_h)


def TradeRow(g, side, model_name, model_prob):
    ask = SideAsk(g, side)
    won = SideWon(g, side)
    last_mid = SideLastMid(g, side)
    edge = (None if model_prob is None else (model_prob - ask - FEE))
    evs = None if model_prob is None else (model_prob - ask - FEE)
    clv = None if last_mid is None else (last_mid - ask)
    return {
        "date": g["game_date"],
        "start_time": g["start_time"],
        "sport": "nba",
        "season": g["season_label"],
        "split": "holdout" if g["eval_split"] == "holdout" else "tune",
        "event_id": g["game_id"],
        "event": g["event"],
        "side": side,
        "model": model_name,
        "model_prob": model_prob,
        "market_prob_devig": g["home_mid"] if side == "home" else g["away_mid"],
        "fill_price": ask,
        "price_basis": PRICE_BASIS,
        "price_time": g["price_time"],
        "fee": FEE,
        "fee_basis": FEE_BASIS,
        "edge": edge,
        "edge_vs_fill": evs,
        "qualifies": bool(edge is not None and edge >= EDGE_CUT),
        "result": won,
        "pnl_at_flat_stake": (100.0 / (ask + FEE)) * won - 100.0,
        "pnl_per_contract": PnlContract(ask, won),
        "clv": clv,
        "abs_move": g.get("abs_move"),
        "dmid_home": g.get("dmid_home"),
        "decile": g.get("move_decile"),
        "cod_under": g.get("cod_under"),
        "_day": g["game_date"],
    }


def SummarizeTrades(trades):
    if not trades:
        return {"n": 0, "hit": None, "mean_pnl": None, "mean_pnl_lo": None, "mean_pnl_hi": None,
                "mean_clv": None, "mean_clv_lo": None, "mean_clv_hi": None, "mean_ask": None,
                "n_days": 0,
                "ci95": {"mean": None, "lo": None, "hi": None, "n": 0, "n_days": 0},
                "ci95_clv": {"mean": None, "lo": None, "hi": None, "n": 0, "n_days": 0}}
    n = len(trades)
    hit = sum(t["result"] for t in trades) / n
    pnls = [(t["_day"], t["pnl_per_contract"]) for t in trades]
    clvs = [(t["_day"], t["clv"]) for t in trades if t.get("clv") is not None]
    ci = BootstrapDayMean(pnls, level=0.95)
    ci_clv = BootstrapDayMean(clvs, level=0.95) if clvs else {"mean": None, "lo": None, "hi": None, "n": 0, "n_days": 0}
    return {
        "n": n,
        "n_days": ci["n_days"],
        "hit": hit,
        "mean_pnl": ci["mean"],
        "mean_pnl_lo": ci["lo"],
        "mean_pnl_hi": ci["hi"],
        "mean_clv": ci_clv["mean"],
        "mean_clv_lo": ci_clv.get("lo"),
        "mean_clv_hi": ci_clv.get("hi"),
        "mean_ask": sum(t["fill_price"] for t in trades) / n,
        "ci95": ci,
        "ci95_clv": ci_clv,
    }


def Bankroll(trades, all_days, day_season, p_for_kelly):
    by_day = defaultdict(list)
    for t in trades:
        by_day[t["date"]].append(t)

    def Run(strategy):
        equity = START_BANK
        peak = START_BANK
        bets_cum = 0
        ruined = False
        ruin_day = None
        rows, stakes_pnls = [], []
        for day in all_days:
            start_of_day = equity
            bets_day = 0
            for t in by_day.get(day, []):
                if ruined or equity < FLAT_STAKE:
                    if not ruined:
                        ruined = True
                        ruin_day = day
                    continue
                fill = t["fill_price"]
                result = t["result"]
                c = fill + FEE
                if strategy == "flat100":
                    stake = FLAT_STAKE
                    if equity < stake:
                        ruined = True
                        ruin_day = day
                        continue
                    pnl = (stake / c) * result - stake
                else:
                    p = p_for_kelly
                    if p is None or c >= 1.0 or c <= 0:
                        kelly = 0.0
                    else:
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
                stakes_pnls.append((stake, pnl))
            if equity > peak:
                peak = equity
            dd = 0.0 if peak == 0 else min(0.0, equity / peak - 1.0)
            season = day_season.get(day, SeasonOf(day))
            split = "holdout" if day >= HOLDOUT_START else "tune"
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
    total = sum(stakes)
    point = sum(pnls) / total if total else None
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
    # 95% to match the holdout request
    return {
        "roi": point,
        "lo": boots[int(0.025 * len(boots))],
        "hi": boots[min(len(boots) - 1, int(0.975 * len(boots)))],
        "n": n,
    }


def Fmt(x, nd=4):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


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


def SelectFade(games, rule, spec):
    """Apply a frozen rule. games should already have move_decile assigned with frozen cuts."""
    out = []
    for g in games:
        if not g.get("has_open") or not g.get("fade_side"):
            continue
        dec = g.get("move_decile")
        if rule == "fade_all_nonzero":
            ok = True
        elif rule == "fade_top_decile":
            ok = dec == 10
        elif rule == "fade_top_two_deciles":
            ok = dec in (9, 10)
        elif rule == "fade_positive_tune_deciles":
            ok = dec in spec["positive_tune_deciles"]
        elif rule == "fade_top_decile_value3c":
            ok = dec == 10
        else:
            ok = False
        if not ok:
            continue
        out.append(g)
    return out


def SelectCOD(games, rule, spec):
    out = []
    cut = spec.get("cod_bottom_cutoff")
    for g in games:
        if not g.get("cod_ready") or not g.get("cod_side"):
            continue
        if rule == "cod_all_min6":
            out.append(g)
        elif rule == "cod_bottom_decile":
            if cut is not None and g["cod_under"] <= cut:
                out.append(g)
    return out


def Main() -> int:
    t0 = time.time()
    os.chdir(MAIN)
    LimitCpu()
    sys.path.insert(0, str(MAIN))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials_path = OUT_DIR / f"{BASE}_trials.jsonl"
    # Start a fresh registry for this job (do not delete prior job folders).
    if trials_path.exists():
        # overwrite by opening 'w' once via a header comment record, then append
        trials_path.write_text("", encoding="utf-8")

    hash_main = GitHash(MAIN)
    print(f"main={hash_main}", flush=True)
    BP = LoadBankrollPlot()

    priced = LoadPricedEdges()
    print(f"priced set {len(priced)} tune={sum(1 for g in priced if g['eval_split']=='tune')} "
          f"holdout={sum(1 for g in priced if g['eval_split']=='holdout')}", flush=True)
    sched, open_by_ticker, ticker_by_key = LoadSchedAndMarkets()
    games, dropped = AttachOpens(priced, sched, open_by_ticker, ticker_by_key)
    games = AttachCOD(games)

    n_true_open_missing = sum(1 for g in games if g.get("true_open_missing"))
    n_has_open = sum(1 for g in games if g.get("has_open"))
    n_zero = sum(1 for g in games if g.get("has_open") and g.get("fade_side") is None)
    print(f"window-open usable={n_has_open} zero-move={n_zero} true_open_missing={n_true_open_missing} "
          f"dropped_open={len(dropped)}", flush=True)

    # ---------- TUNE ONLY: cutoffs and decile diagnostics ----------
    tune = [g for g in games if g["eval_split"] == "tune"]
    holdout_games = [g for g in games if g["eval_split"] == "holdout"]
    tune_moves = [g for g in tune if g.get("has_open") and g.get("fade_side")]
    # Rank-based deciles on tune (equal count; ties broken by game_id). Tick size is
    # one cent, so value-quantile cuts would collapse and leave empty bins.
    ranked = sorted(tune_moves, key=lambda g: (g["abs_move"], g["game_id"]))
    n_tm = len(ranked)
    for i, g in enumerate(ranked):
        g["move_decile"] = min(10, i * 10 // n_tm + 1) if n_tm else None
    # Frozen cutoffs for holdout: minimum |move| observed in each tune decile.
    decile_min = {}
    decile_max = {}
    for d in range(1, 11):
        xs = [g["abs_move"] for g in ranked if g["move_decile"] == d]
        decile_min[d] = min(xs) if xs else None
        decile_max[d] = max(xs) if xs else None
    cutoff_top = decile_min.get(10)
    cutoff_top_two = decile_min.get(9)
    move_cuts = [decile_min[d] for d in range(2, 11) if decile_min.get(d) is not None]
    print("tune nonzero-move", n_tm, "decile_min", decile_min, "top_cutoff", cutoff_top, flush=True)

    def HoldoutDecile(abs_move):
        if cutoff_top is not None and abs_move >= cutoff_top:
            return 10
        # Walk high to low so a value that sits on a tied boundary goes to the higher decile.
        assigned = 1
        for d in range(1, 10):
            lo = decile_min.get(d)
            if lo is not None and abs_move >= lo:
                assigned = d
        return assigned

    for g in games:
        if g.get("eval_split") == "tune" and g.get("has_open") and g.get("fade_side"):
            # already set on ranked tune_moves (same objects)
            if g.get("move_decile") is None:
                g["move_decile"] = HoldoutDecile(g["abs_move"])
        elif g.get("has_open") and g.get("fade_side"):
            g["move_decile"] = HoldoutDecile(g["abs_move"])
        else:
            g["move_decile"] = None

    # Tune COD cutoff: 10th percentile of underperformer residual (ready games only).
    tune_cod = [g for g in tune if g.get("cod_ready") and g.get("cod_side")]
    cod_cuts = PercentileCuts([g["cod_under"] for g in tune_cod], 10) if tune_cod else []
    cod_bottom_cutoff = cod_cuts[0] if cod_cuts else None  # 10th percentile
    for g in games:
        if g.get("cod_ready") and g.get("cod_under") is not None and cod_bottom_cutoff is not None:
            g["cod_decile"] = 1 if g["cod_under"] <= cod_bottom_cutoff else None
        else:
            g["cod_decile"] = None

    # Per-decile tune table (move size vs outcome). No holdout peek.
    decile_rows = []
    positive_deciles = []
    for d in range(1, 11):
        subset = [g for g in tune_moves if g["move_decile"] == d]
        trades = [TradeRow(g, g["fade_side"], "fade_decile", None) for g in subset]
        sm = SummarizeTrades(trades)
        mean_abs = (sum(g["abs_move"] for g in subset) / len(subset)) if subset else None
        mean_signed = (sum(g["dmid_home"] for g in subset) / len(subset)) if subset else None
        riser_hit = None
        if subset:
            riser_hit = sum(SideWon(g, g["rise_side"]) for g in subset) / len(subset)
        rec = {
            "decile": d, "split": "tune", "n": sm["n"],
            "cut_lo": (min(g["abs_move"] for g in subset) if subset else None),
            "cut_hi": (max(g["abs_move"] for g in subset) if subset else None),
            "mean_abs_move": mean_abs, "mean_dmid_home": mean_signed,
            "riser_hit_rate": riser_hit, "fade_hit_rate": sm["hit"],
            "mean_pnl_after_fee": sm["mean_pnl"],
            "mean_pnl_lo95": sm["mean_pnl_lo"], "mean_pnl_hi95": sm["mean_pnl_hi"],
            "mean_clv": sm["mean_clv"],
        }
        decile_rows.append(rec)
        if sm["n"] and sm["mean_pnl"] is not None and sm["mean_pnl"] > 0:
            positive_deciles.append(d)
        AppendTrial(trials_path, {
            "trial_id": f"tune_fade_decile_{d}",
            "family": "fade_decile_diagnostic",
            "params": {"decile": d, "move_cuts": move_cuts},
            "control": False, "split_scored": "tune", "descriptive": True,
            "metrics": rec,
        })
        print(f"tune D{d} n={sm['n']} |move|={Fmt(mean_abs,4)} riser_hit={Fmt(riser_hit,3)} "
              f"fade_pnl={Fmt(sm['mean_pnl'],4)}", flush=True)

    # Temporary spec to evaluate tune metrics for freezing p_tune / selected deciles.
    draft_spec = {
        "positive_tune_deciles": positive_deciles,
        "cod_bottom_cutoff": cod_bottom_cutoff,
        "move_cuts": move_cuts,
    }

    def TuneMetrics(rule, selector, side_fn, p_field_later=None):
        chosen = selector(tune, rule, draft_spec)
        trades = [TradeRow(g, side_fn(g), rule, None) for g in chosen if side_fn(g)]
        return chosen, trades, SummarizeTrades(trades)

    tune_store = {}
    for rule in TRIALS_PRE_REG:
        if rule.startswith("cod_"):
            chosen, trades, sm = TuneMetrics(rule, SelectCOD, lambda g: g.get("cod_side"))
        elif rule == "fade_top_decile_value3c":
            # p will be the top-decile tune hit rate; apply after we have it.
            continue
        else:
            chosen, trades, sm = TuneMetrics(rule, SelectFade, lambda g: g.get("fade_side"))
        tune_store[rule] = {"summary": sm, "n_candidates": len(chosen)}
        AppendTrial(trials_path, {
            "trial_id": f"tune_{rule}",
            "family": rule, "params": {"fitted_on": "tune_through_2026-01-31"},
            "control": rule in ("fade_all_nonzero", "cod_all_min6"),
            "split_scored": "tune", "descriptive": False,
            "metrics": sm,
        })
        print(f"tune {rule}: n={sm['n']} hit={Fmt(sm['hit'],3)} pnl={Fmt(sm['mean_pnl'],4)} "
              f"CI=[{Fmt(sm['mean_pnl_lo'],4)}, {Fmt(sm['mean_pnl_hi'],4)}]", flush=True)

    p_primary = (tune_store[PRIMARY]["summary"]["hit"] if PRIMARY in tune_store else None)
    # 3c overlay on tune, using frozen-to-be primary hit rate as p.
    if p_primary is not None:
        chosen = SelectFade(tune, "fade_top_decile", draft_spec)
        trades = []
        for g in chosen:
            ask = SideAsk(g, g["fade_side"])
            if p_primary - ask - FEE >= EDGE_CUT:
                trades.append(TradeRow(g, g["fade_side"], "fade_top_decile_value3c", p_primary))
        sm = SummarizeTrades(trades)
        tune_store["fade_top_decile_value3c"] = {"summary": sm, "n_candidates": len(chosen)}
        AppendTrial(trials_path, {
            "trial_id": "tune_fade_top_decile_value3c",
            "family": "fade_top_decile_value3c",
            "params": {"p_tune": p_primary, "edge_cut": EDGE_CUT},
            "control": False, "split_scored": "tune", "descriptive": False,
            "metrics": sm,
        })
        print(f"tune fade_top_decile_value3c: n={sm['n']} pnl={Fmt(sm['mean_pnl'],4)}", flush=True)

    p_by_rule = {rule: (tune_store[rule]["summary"]["hit"] if rule in tune_store else None)
                 for rule in TRIALS_PRE_REG}

    # ---------- FREEZE: write spec + hash BEFORE any holdout metric ----------
    frozen = {
        "name": "nba_overreaction_fade_v1",
        "pre_registered_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hypothesis": "Fade the Kalshi side whose price rose the most from the first pulled candle to T-60. Primary rule uses the top tune decile of |dmid|. Variant backs the team that has underperformed its own prior Kalshi-implied expectation (COD).",
        "priced_set": "2026-11-13_calibration / bankroll-10k kalshi_T60, 1469 games",
        "tune_end_inclusive": TUNE_END,
        "holdout_start_inclusive": HOLDOUT_START,
        "fee": FEE,
        "fee_basis": FEE_BASIS,
        "edge_cut": EDGE_CUT,
        "open_definition": "First valid 1-minute candle at or before T-60 in the pulled candle file (window-open). Listed market open is recorded but is generally not in the pull.",
        "t60_definition": "Last valid 1-minute candle at or before scheduled start minus 60 minutes; drop if older than 30 minutes.",
        "clv_definition": "Last pregame mid (last valid candle at or before scheduled tip) for the bet side, minus T-60 ask paid.",
        "away_ask": "1 - home_bid (home YES candles only)",
        "cod_definition": "For each team, sum(result - kalshi_T60_mid) over prior priced games in the same season. Back the more negative residual. Require at least 6 prior priced games per team.",
        "cod_min_prior": COD_MIN_PRIOR,
        "move_decile_cuts": move_cuts,
        "tune_decile_min_abs_move": decile_min,
        "tune_decile_max_abs_move": decile_max,
        "cutoff_top_decile": cutoff_top,
        "cutoff_top_two_deciles": cutoff_top_two,
        "positive_tune_deciles": positive_deciles,
        "cod_bottom_cutoff": cod_bottom_cutoff,
        "p_tune_hit_rate": p_by_rule,
        "primary_rule": PRIMARY,
        "pre_registered_trials": list(TRIALS_PRE_REG),
        "n_tune_priced": len(tune),
        "n_tune_nonzero_move": len(tune_moves),
        "n_tune_cod_ready": len(tune_cod),
        "code_head": hash_main,
        "verdict_rule": "PASS unless the primary fade_top_decile holdout mean P/L per contract after Tim's $0.02 fee has a 95% day-block bootstrap CI entirely above 0.",
    }
    spec_path = OUT_DIR / f"{BASE}_spec.json"
    sha_path = OUT_DIR / f"{BASE}_spec.sha256"
    # If a spec was already frozen (e.g. a later crash after the holdout peek), reuse it.
    # Do not retune cutoffs after any holdout numbers have been seen.
    if spec_path.exists() and sha_path.exists():
        frozen = json.loads(spec_path.read_text(encoding="utf-8"))
        frozen_hash = sha_path.read_text(encoding="utf-8").strip()
        stored = frozen.get("spec_sha256")
        print(f"REUSING FROZEN SPEC {spec_path} sha256={frozen_hash} stored={stored}", flush=True)
    else:
        frozen_hash = CanonicalHash(frozen)
        frozen["spec_sha256"] = frozen_hash
        spec_path.write_text(json.dumps(frozen, indent=2, default=str), encoding="utf-8")
        sha_path.write_text(frozen_hash + "\n", encoding="utf-8")
        print(f"FROZEN SPEC {spec_path} sha256={frozen_hash}", flush=True)
    AppendTrial(trials_path, {
        "trial_id": "frozen_spec",
        "family": "spec",
        "params": {"spec_sha256": frozen_hash, "primary_rule": PRIMARY},
        "split_scored": "none",
        "descriptive": False,
        "metrics": {"spec_written_before_holdout": True},
    })

    # ---------- HOLDOUT (uses frozen cutoffs only) ----------
    holdout_results = {}
    holdout_trades = {}
    for rule in TRIALS_PRE_REG:
        if rule.startswith("cod_"):
            chosen = SelectCOD(holdout_games, rule, frozen)
            side_fn = lambda g: g.get("cod_side")
        else:
            chosen = SelectFade(holdout_games, rule, frozen)
            side_fn = lambda g: g.get("fade_side")
        p = p_by_rule.get(rule)
        trades = []
        for g in chosen:
            side = side_fn(g)
            if not side:
                continue
            if rule == "fade_top_decile_value3c":
                ask = SideAsk(g, side)
                if p is None or p - ask - FEE < EDGE_CUT:
                    continue
            trades.append(TradeRow(g, side, rule, p))
        sm = SummarizeTrades(trades)
        holdout_results[rule] = sm
        holdout_trades[rule] = trades
        AppendTrial(trials_path, {
            "trial_id": f"holdout_{rule}",
            "family": rule,
            "params": {"spec_sha256": frozen_hash, "p_tune": p},
            "control": rule in ("fade_all_nonzero", "cod_all_min6"),
            "split_scored": "holdout",
            "descriptive": False,
            "metrics": sm,
        })
        print(f"holdout {rule}: n={sm['n']} hit={Fmt(sm['hit'],3)} pnl={Fmt(sm['mean_pnl'],4)} "
              f"CI=[{Fmt(sm['mean_pnl_lo'],4)}, {Fmt(sm['mean_pnl_hi'],4)}] clv={Fmt(sm['mean_clv'],4)}", flush=True)

    # Bankrolls on the holdout, primary rule, starting $10k at the first holdout game day.
    primary_trades = holdout_trades[PRIMARY]
    ho_days = sorted({g["game_date"] for g in holdout_games})
    day_season = {}
    for g in holdout_games:
        day_season.setdefault(g["game_date"], g["season_label"])
    paths = Bankroll(primary_trades, ho_days, day_season, p_by_rule.get(PRIMARY))
    # Also bankroll the COD variant (separate strategies).
    paths_cod = Bankroll(holdout_trades["cod_bottom_decile"], ho_days, day_season,
                         p_by_rule.get("cod_bottom_decile"))

    bank_rows = []
    def frame_paths(path_map, suffix=""):
        for name, payload in path_map.items():
            dates = [r["date"] for r in payload["rows"]]
            equity = [r["equity"] for r in payload["rows"]]
            frame = BP.FrameBankroll(dates, equity, series=name, mode="index")
            xmap = {row["date"]: int(row["x"]) + 1 for row in frame["rows"]}
            ddmap = {row["date"]: row["drawdown_pct"] for row in frame["rows"]}
            strat = name + suffix
            for r in payload["rows"]:
                rec = dict(r)
                rec["strategy"] = strat
                rec["series"] = f"nba_{strat}_{PRICE_BASIS}"
                rec["x"] = xmap[r["date"]]
                rec["drawdown_pct"] = ddmap[r["date"]]
                bank_rows.append(rec)
    frame_paths(paths, "")
    frame_paths(paths_cod, "_cod")
    bank_rows.sort(key=lambda r: (r["x"], r["strategy"]))

    roi = {name: BootstrapRoi(paths[name]["stakes_pnls"]) for name in paths}
    roi_cod = {name: BootstrapRoi(paths_cod[name]["stakes_pnls"]) for name in paths_cod}

    # Edges CSV: primary holdout trades, 21 plan columns (plus we may carry extras ignored if we only write plan fields).
    edge_fields = ["date", "start_time", "sport", "season", "split", "event_id", "event", "side",
                   "model", "model_prob", "market_prob_devig", "fill_price", "price_basis",
                   "price_time", "fee", "fee_basis", "edge", "edge_vs_fill", "qualifies",
                   "result", "pnl_at_flat_stake"]
    bank_fields = ["date", "sport", "series", "strategy", "price_basis", "season", "split",
                   "x", "bets_day", "bets_cum", "equity", "drawdown_pct"]
    WriteCsv(OUT_DIR / f"{BASE}.csv", bank_rows, bank_fields)
    WriteCsv(OUT_DIR / f"{BASE}_edges.csv", primary_trades, edge_fields)
    WriteCsv(OUT_DIR / f"{BASE}_tune_deciles.csv", decile_rows,
             ["decile", "split", "n", "cut_lo", "cut_hi", "mean_abs_move", "mean_dmid_home",
              "riser_hit_rate", "fade_hit_rate", "mean_pnl_after_fee", "mean_pnl_lo95",
              "mean_pnl_hi95", "mean_clv"])

    # Holdout table csv
    ho_rows = []
    for rule in TRIALS_PRE_REG:
        sm = holdout_results[rule]
        ho_rows.append({
            "rule": rule, "split": "holdout", "primary": rule == PRIMARY,
            "n": sm["n"], "n_days": sm.get("n_days"),
            "hit_rate": sm["hit"],
            "fair_value": p_by_rule.get(rule),
            "mean_pnl_after_fee": sm["mean_pnl"],
            "ci95_lo": sm["mean_pnl_lo"], "ci95_hi": sm["mean_pnl_hi"],
            "mean_clv": sm["mean_clv"],
            "mean_clv_lo": sm.get("mean_clv_lo"), "mean_clv_hi": sm.get("mean_clv_hi"),
            "mean_ask": sm.get("mean_ask"),
        })
    WriteCsv(OUT_DIR / f"{BASE}_holdout.csv", ho_rows,
             ["rule", "split", "primary", "n", "n_days", "hit_rate", "fair_value",
              "mean_pnl_after_fee", "ci95_lo", "ci95_hi", "mean_clv", "mean_clv_lo",
              "mean_clv_hi", "mean_ask"])

    # Verdict on the primary only.
    prim = holdout_results[PRIMARY]
    ci_ok = prim["mean_pnl"] is not None and prim["mean_pnl_lo"] is not None and prim["mean_pnl_lo"] > 0
    verdict_pass = not ci_ok

    # ---------- markdown ----------
    lines, a = [], (lambda s: lines.append(s))
    a("# NBA overreaction fade vs Kalshi T-60 (2026-11-13)")
    a("")
    a("Simulation only. No wagers were placed. Default call is PASS unless the pre-registered primary rule's holdout 95% day-block CI on mean P/L after fee is entirely above zero.")
    a("")
    a("## Pre-registered hypothesis")
    a("- Fade the side whose Kalshi price rose the most from the **first pulled 1-minute candle** to T-60.")
    a("- Primary rule: trade only when `|Δmid|` is in the **top decile** of the tune distribution.")
    a("- Variant: back the team that has underperformed its own prior Kalshi-implied expectation (COD residual = sum(result − T-60 mid) over earlier priced games this season, minimum 6 prior games). Trade the bottom tune decile of the underperformer residual.")
    a(f"- Tune: priced games through {TUNE_END} (N={len(tune)}). Holdout: {HOLDOUT_START} through the end of 2025-26, plus 2026-27 to date (N={len(holdout_games)}).")
    a(f"- Frozen spec hash: `{frozen_hash}`. The spec file was written and hashed **before** any holdout metric was computed.")
    a(f"- Fee: Tim's $0.02 working estimate (`{FEE_BASIS}`), not a verified exchange fee. Edge = p − (ask + 0.02); a 3¢ bar is used only in the value overlay trial. The fade itself has no model probability; the fair value below is the **frozen tune hit rate** of each rule, and the headline holdout number is mean P/L per contract after the fee.")
    a(f"- Code: `{hash_main}` (local).")
    a("")
    a("## Open-candle coverage (important)")
    a(f"- Priced set: {len(games)} games. Window-open + T-60 usable: {n_has_open}. Zero move (no fade side): {n_zero}. Dropped before a move could be measured: {len(dropped)} ({', '.join(f'{k}={v}' for k,v in Counter(d['reason'] for d in dropped).most_common()) or 'none'}).")
    a(f"- **True listed opens are missing from the pull.** {n_true_open_missing} of {n_has_open} usable games have a first candle more than one hour after `markets.json` open_time. In this pull the first candle is always within about two hours of T-60 (typical gap ~75 minutes). This is not an open-to-T-60 move; it is late-pregame drift over the pulled window. Every result below is conditional on that window.")
    a("- Only home YES candles were pulled. Away mid/ask are implied (`1 - home mid`, `1 - home bid`). Candles have no size.")
    a("")
    a("## Tune deciles (move size vs outcome)")
    a("")
    a("Fitted on tune only. `riser_hit_rate` is how often the side that rose also won. Fade P/L is per contract after Tim's $0.02 fee.")
    a("")
    a("| decile | N | mean |Δmid| | riser hit | fade hit | fade mean P/L after fee [95% day CI] | mean CLV |")
    a("|---:|---:|---:|---:|---:|---|---:|")
    for rec in decile_rows:
        a(f"| {rec['decile']} | {rec['n']} | {Fmt(rec['mean_abs_move'],4)} | {Fmt(rec['riser_hit_rate'],3)} | {Fmt(rec['fade_hit_rate'],3)} | {Fmt(rec['mean_pnl_after_fee'],4)} [{Fmt(rec['mean_pnl_lo95'],4)}, {Fmt(rec['mean_pnl_hi95'],4)}] | {Fmt(rec['mean_clv'],4)} |")
    a("")
    a(f"- Tune deciles with mean P/L after fee > 0 (fitted set, then frozen): {positive_deciles or 'none'}.")
    a(f"- Primary fair value (tune hit rate of `{PRIMARY}`): {FairDollars(p_primary)}.")
    a(f"- COD bottom-decile cutoff (tune 10th percentile of the underperformer residual): {Fmt(cod_bottom_cutoff, 4)}. Tune COD-ready games: {len(tune_cod)}.")
    a("")
    a("## Holdout")
    a("")
    a("Cutoffs were frozen before this table was computed. 95% CIs are day-block bootstraps (10,000 reps). Fair value is the frozen tune hit rate, shown as a contract price.")
    a("")
    a("| rule | primary | N | N days | fair value | hit | mean P/L / contract after $0.02 [95% CI] | mean CLV [95% CI] |")
    a("|---|---|---:|---:|---:|---:|---|---|")
    for rule in TRIALS_PRE_REG:
        sm = holdout_results[rule]
        star = "yes" if rule == PRIMARY else ""
        a(f"| {rule} | {star} | {sm['n']} | {sm.get('n_days') or 0} | {FairDollars(p_by_rule.get(rule))} | {Fmt(sm['hit'],3)} | {Fmt(sm['mean_pnl'],4)} [{Fmt(sm['mean_pnl_lo'],4)}, {Fmt(sm['mean_pnl_hi'],4)}] | {Fmt(sm['mean_clv'],4)} [{Fmt(sm.get('mean_clv_lo'),4)}, {Fmt(sm.get('mean_clv_hi'),4)}] |")
    a("")
    a("## $10k bankrolls (holdout only, primary rule)")
    a("")
    a("Start $10,000 on the first holdout game day, no reset. `flat100` always stakes $100 on a primary-rule fade (stop if equity < $100). `qkelly` uses the frozen tune hit rate as p, 0.25× Kelly, 2% of start-of-day equity cap. Charts use `Backtest.BankrollPlot` (season index, summer skipped). The COD variant is in the same bankroll file as `flat100_cod` / `qkelly_cod`.")
    a("")
    a("| strategy | N bets | ending bankroll | ROI per $ staked [95% event bootstrap CI] | max drawdown | ruined |")
    a("|---|---:|---:|---|---:|---|")
    for label, pth, rmap in (("flat100", paths, roi), ("qkelly", paths, roi),
                             ("flat100_cod", paths_cod, roi_cod), ("qkelly_cod", paths_cod, roi_cod)):
        key = label.replace("_cod", "")
        p = pth[key]
        r = rmap[key]
        roi_s = "n/a" if r["roi"] is None else f"{r['roi']:+.3f} [{Fmt(r['lo'], 3)}, {Fmt(r['hi'], 3)}]"
        a(f"| {label} | {p['n_bets']} | ${p['end']:,.2f} | {roi_s} | {p['max_dd']:.3f} | {'yes at ' + p['ruin_day'] if p['ruined'] else 'no'} |")
    a("")
    a("## Caveats")
    a("- **Open-candle sparsity / missing opens:** the Kalshi candle pull does not include the listed market open. First candles sit about 1-2 hours before T-60 and typically 1-2+ days after `open_time`. This test cannot speak to open-to-T-60 overreaction; only late-pregame drift in the pulled window.")
    a("- Moves are small: tune |Δmid| is often a few cents (many ties at a cent tick). Top-decile is a short-horizon drift, not a day-long march.")
    a("- Away prices are implied from the home YES book. No size; T-60 asks are optimistic for large stakes.")
    a("- The $0.02 fee is Tim's working estimate, not a verified Kalshi taker fee.")
    a("- The fade has no model probability. Fair value is the frozen tune hit rate, reused on the holdout; it is not a walk-forward probability.")
    a("- COD uses only prior *priced* games in the same season (games without a T-60 mid never enter the residual). Early-season games are excluded by the 6-game minimum.")
    a("- Every trial we scored is in the trial registry, including tune decile diagnostics and control rules. The verdict uses only the pre-registered primary rule.")
    a("- No injury adjustment (this is a price-only rule).")
    a("")
    if verdict_pass:
        a(f"- **Verdict: PASS.** The primary `{PRIMARY}` holdout mean P/L after Tim's $0.02 fee does not have a 95% day-block CI entirely above zero. Do not bet this fade.")
    else:
        a(f"- The primary `{PRIMARY}` holdout 95% CI is above zero on this sample. Treat as research only until Tim says otherwise; this is not a live bet, and the missing true opens remain a serious caveat.")
    a("")
    (OUT_DIR / f"{BASE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        d, e, s = [], [], []
        for name in ("flat100", "qkelly"):
            for r in paths[name]["rows"]:
                d.append(r["date"]); e.append(r["equity"]); s.append(name)
        frame = BP.FrameBankroll(d, e, s, mode="index")
        BP.PlotBankroll(frame, path=str(OUT_DIR / f"{BASE}_draft.png"),
                        title="NBA $10k overreaction fade (holdout, summer skipped)",
                        ylabel="Simulated bankroll (US dollars)")
        print("wrote draft png", flush=True)
    except Exception as exc:
        print(f"draft plot skipped: {exc}", flush=True)

    dump = {
        "hash_main": hash_main,
        "spec_sha256": frozen_hash,
        "n_priced": len(games), "n_tune": len(tune), "n_holdout": len(holdout_games),
        "n_has_open": n_has_open, "n_zero_move": n_zero,
        "n_true_open_missing": n_true_open_missing,
        "dropped": dict(Counter(d["reason"] for d in dropped)),
        "move_cuts": move_cuts, "positive_tune_deciles": positive_deciles,
        "cod_bottom_cutoff": cod_bottom_cutoff,
        "p_tune": p_by_rule,
        "fair_value_primary": p_primary,
        "tune": {k: v["summary"] for k, v in tune_store.items()},
        "holdout": holdout_results,
        "bankroll_primary": {k: {"end": v["end"], "n_bets": v["n_bets"], "max_dd": v["max_dd"],
                                 "ruined": v["ruined"], "ruin_day": v["ruin_day"], "roi": roi[k]}
                             for k, v in paths.items()},
        "bankroll_cod": {k: {"end": v["end"], "n_bets": v["n_bets"], "max_dd": v["max_dd"],
                             "ruined": v["ruined"], "ruin_day": v["ruin_day"], "roi": roi_cod[k]}
                         for k, v in paths_cod.items()},
        "verdict_pass": verdict_pass,
        "elapsed_s": time.time() - t0,
    }
    (OUT_DIR / f"{BASE}_run.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: dump[k] for k in dump if k not in ("tune", "holdout")}, indent=2, default=str), flush=True)
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
        Path(r"E:\Basketball\basketball\logs\2026-11-13_overreaction-fade.exc").write_text(tb, encoding="utf-8")
        raise