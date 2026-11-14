"""Score the preregistered Kalshi player-prop T-60 backtest. Never changes the frozen spec."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from Props.Model import (
    TRAIN_SEASON, AttachCountPredictions, AttachPredictions, BuildFeatures,
    FairGe, FilterCalib, FitCountParams, FitParams, NbGe, PoissonGe,
)
from Props.PullProps import (
    ROOT, TARGET_SERIES, QuoteAtT60, IsoToTs, TipTs, Log,
)
from InPlay.Http import ReadJson
from Backtest.ResidualSearch import DayBlockBootstrap, HansenSpa, MonteCarlo

import numpy as np

FEE = 0.02
FEE_BASIS = "tim_est_0.02"
EDGE_CUT = 0.03
EVAL_START = "2025-04-14"
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_props-backtest")
BASE = "2026-11-13_props-backtest"
PREREG_HASH = "9b5fb1fe4b4de7401144e0b56fd8743af2387555"
FAMILY_OF = {
    "KXNBA3PT": "3pm", "KXNBA3PTPROT": "3pm",
    "KXNBAPTS": "pts", "KXNBAPTSPROT": "pts",
    "KXNBAREB": "reb", "KXNBAREBPROT": "reb",
    "KXNBAAST": "ast", "KXNBAASTPROT": "ast",
    "KXNBAPRA": "pra", "KXNBAPRAPROT": "pra",
}
NORMAL_PRIORITY_CLASS = 0x00000020


def LowerToNormal():
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def WriteJson(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def WriteCsv(path, rows, fields=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def GitHash():
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def Fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def NormName(s):
    return " ".join(str(s or "").lower().replace(".", "").replace("'", "").split())


def StrikeBucket(k):
    if k is None:
        return "unknown"
    k = int(k)
    return str(k) if k <= 5 else "6plus"


def LoadSeasons():
    from Props.ExportThrees import LoadPlayerGames, LoadSeasonGames
    players = LoadPlayerGames([2022, 2023, 2024, 2025, 2026])
    games = LoadSeasonGames([2022, 2023, 2024, 2025, 2026])
    return players, games


def BuildIndex(rows):
    by_name = defaultdict(list)
    by_last = defaultdict(list)
    for r in rows:
        key = (r["date"], NormName(r["player_name"]))
        by_name[key].append(r)
        parts = NormName(r["player_name"]).split()
        if parts:
            by_last[(r["date"], parts[-1])].append(r)
    return by_name, by_last


def MatchRow(market, by_name, by_last):
    date = market.get("date")
    name = NormName(market.get("player_name"))
    if not date or not name or date < EVAL_START:
        return None, "no_date_or_name"
    rows = by_name.get((date, name)) or []
    if len(rows) == 1:
        return rows[0], "full_name"
    if len(rows) > 1:
        team = market.get("away") if market.get("away") else None
        # Prefer the team abbreviation if it matches home/away and the player's team.
        sides = {market.get("away"), market.get("home")}
        hit = [r for r in rows if r.get("team") in sides]
        if len(hit) == 1:
            return hit[0], "full_name_team"
        return None, "ambiguous_full_name"
    last = name.split()[-1] if name.split() else ""
    cands = by_last.get((date, last)) or []
    if len(cands) == 1:
        return cands[0], "last_name_unique"
    if len(cands) > 1:
        sides = {market.get("away"), market.get("home")}
        hit = [r for r in cands if r.get("team") in sides]
        if len(hit) == 1:
            return hit[0], "last_name_team"
        return None, "ambiguous_last_name"
    return None, "no_player_game"


def LoadMarketsAndQuotes():
    """Join cached markets to T-60 quotes. Does not fetch."""
    candle_dir = ROOT / "candles"
    rows = []
    coverage = []
    for series in TARGET_SERIES:
        path = ROOT / f"markets_{series}.json"
        if not path.exists():
            coverage.append({"series": series, "n_markets": 0, "n_quoted": 0, "note": "no_list"})
            continue
        markets = ReadJson(str(path))
        n_result = n_eval = n_quoted = n_candle_files = n_empty = 0
        dates = []
        for m in markets:
            if m.get("result") not in ("yes", "no", "Yes", "No"):
                continue
            n_result += 1
            if not m.get("date") or m["date"] < EVAL_START:
                continue
            n_eval += 1
            if m.get("date"):
                dates.append(m["date"])
            cpath = candle_dir / (m["ticker"] + ".json")
            quote = None
            candle_src = None
            tip_ts, tip_src = TipTs(m)
            if cpath.exists():
                n_candle_files += 1
                blob = ReadJson(str(cpath))
                candles = blob.get("candles") or []
                candle_src = blob.get("source")
                if not candles:
                    n_empty += 1
                if tip_ts and candles:
                    quote = QuoteAtT60(candles, tip_ts)
            rec = dict(m)
            rec["family"] = FAMILY_OF.get(series) or m.get("family")
            rec["tip_ts"] = tip_ts
            rec["tip_src"] = tip_src
            rec["candle_src"] = candle_src
            rec["quote"] = quote
            rec["quoted"] = bool(quote and quote.get("ask") is not None)
            rec["yes_result"] = 1 if str(m["result"]).lower() == "yes" else 0
            rows.append(rec)
            if rec["quoted"]:
                n_quoted += 1
        coverage.append({
            "series": series, "n_markets": len(markets), "n_with_result": n_result,
            "n_eval": n_eval, "n_candle_files": n_candle_files, "n_empty_candles": n_empty,
            "n_quoted_t60": n_quoted,
            "date_min": min(dates) if dates else None, "date_max": max(dates) if dates else None,
        })
    return rows, coverage


def BinaryLL(p, y):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    y = float(y)
    return - (y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def ScoreUniverse(rows, by_name, by_last, params_by_family):
    scored, skipped = [], defaultdict(int)
    for m in rows:
        if not m.get("quoted"):
            skipped["no_t60_quote"] += 1
            continue
        family = m.get("family")
        k = m.get("k")
        if family not in params_by_family or not k:
            skipped["no_family_or_k"] += 1
            continue
        prow, how = MatchRow(m, by_name, by_last)
        if prow is None:
            skipped[how] += 1
            continue
        p = FairGe(prow, family, int(k), params_by_family[family])
        if p is None:
            skipped["no_fair"] += 1
            continue
        q = m["quote"]
        yes_ask, yes_bid, mid = q.get("ask"), q.get("bid"), q.get("mid")
        implied = False
        no_ask = None
        if q.get("no_ask") is not None:
            no_ask = float(q["no_ask"])
        elif yes_bid is not None:
            no_ask = 1.0 - float(yes_bid)
            implied = True
        e_yes = None if yes_ask is None else float(p) - (float(yes_ask) + FEE)
        e_no = None if no_ask is None else (1.0 - float(p)) - (float(no_ask) + FEE)
        side = None
        edge = None
        fill = None
        if e_yes is not None and e_no is not None:
            if e_yes >= e_no and e_yes >= EDGE_CUT:
                side, edge, fill = "yes", e_yes, float(yes_ask)
            elif e_no > e_yes and e_no >= EDGE_CUT:
                side, edge, fill = "no", e_no, float(no_ask)
        elif e_yes is not None and e_yes >= EDGE_CUT:
            side, edge, fill = "yes", e_yes, float(yes_ask)
        elif e_no is not None and e_no >= EDGE_CUT:
            side, edge, fill = "no", e_no, float(no_ask)
        y_yes = m["yes_result"]
        result = y_yes if side == "yes" else (1 - y_yes if side == "no" else None)
        pnl = None if side is None else result - (fill + FEE)
        ll_model = BinaryLL(p, y_yes)
        ll_mid = BinaryLL(mid if mid is not None else yes_ask, y_yes) if (mid is not None or yes_ask is not None) else None
        scored.append({
            "date": m["date"], "series": m["series"], "family": family, "ticker": m["ticker"],
            "player_name": m.get("player_name"), "k": int(k), "strike_bucket": StrikeBucket(k),
            "model_p": p, "yes_ask": yes_ask, "yes_bid": yes_bid, "mid": mid,
            "no_ask": no_ask, "implied_no_ask": implied, "spread": q.get("spread"),
            "ask_size": q.get("ask_size"), "bid_size": q.get("bid_size"),
            "quote_t": q.get("t"), "quote_age_s": q.get("age_s"), "tip_src": m.get("tip_src"),
            "candle_src": m.get("candle_src"), "match": how,
            "yes_result": y_yes, "ll_model": ll_model, "ll_mid": ll_mid,
            "side": side, "edge": edge, "fill": fill, "result": result, "pnl": pnl,
            "qualifies": side is not None, "p": (p if side == "yes" else (1.0 - p if side == "no" else None)),
            "fee": FEE, "fee_basis": FEE_BASIS, "protected": bool(m.get("protected")),
            "prior_n": prow.get("prior_n"), "proj_min": prow.get("proj_min"),
        })
        skipped["scored"] += 1
    return scored, dict(skipped)


def SpaAndCi(scored):
    bets = [r for r in scored if r["qualifies"]]
    hold = None
    if bets:
        hold = DayBlockBootstrap([b["pnl"] for b in bets], [b["date"] for b in bets], reps=10000, seed=7, level=0.95)
        hold["n_bets"] = len(bets)
        hold["n_days"] = len({b["date"] for b in bets})
        hold["wins"] = int(sum(b["result"] for b in bets))
    # Log-loss gap in bits, all quoted scored markets with a mid.
    gap = None
    with_mid = [r for r in scored if r.get("ll_mid") is not None]
    if with_mid:
        bits = [(r["ll_mid"] - r["ll_model"]) / math.log(2.0) for r in with_mid]
        gap = DayBlockBootstrap(bits, [r["date"] for r in with_mid], reps=10000, seed=7, level=0.90)
        gap["n"] = len(with_mid)
    # SPA: one column per (series, strike_bucket, side), rows = eval dates
    # Daily mean pnl for that rule; 0 if no bet that day.
    rules = sorted({(r["series"], r["strike_bucket"], side) for r in scored for side in ("yes", "no")})
    dates = sorted({r["date"] for r in scored})
    date_i = {d: i for i, d in enumerate(dates)}
    K = len(rules)
    T = len(dates)
    # Sum and count per (date, rule) then mean; 0 if no bets.
    sums = np.zeros((T, K), dtype=np.float64)
    cnts = np.zeros((T, K), dtype=np.int64)
    rule_ids = []
    for j, (series, bucket, side) in enumerate(rules):
        rule_ids.append(f"{series}_{bucket}_{side}")
        for b in bets:
            if b["series"] == series and b["strike_bucket"] == bucket and b["side"] == side:
                i = date_i[b["date"]]
                sums[i, j] += b["pnl"]
                cnts[i, j] += 1
    D = np.zeros((T, K), dtype=np.float64)
    for j in range(K):
        for i in range(T):
            D[i, j] = (sums[i, j] / cnts[i, j]) if cnts[i, j] else 0.0
    spa = HansenSpa(D, reps=10000, seed=7, mean_block=5) if T >= 2 and K >= 1 else None
    return bets, hold, gap, spa, rule_ids, D


def Spreads(scored):
    out = []
    by = defaultdict(list)
    for r in scored:
        by[r["series"]].append(r)
    for series, recs in sorted(by.items()):
        spreads = [r["spread"] for r in recs if r.get("spread") is not None]
        depths_ask = [r["ask_size"] for r in recs if r.get("ask_size") is not None]
        depths_bid = [r["bid_size"] for r in recs if r.get("bid_size") is not None]
        out.append({
            "date": "2026-11-13", "sport": "nba", "series": series,
            "n_quoted": len(recs),
            "mean_spread": float(np.mean(spreads)) if spreads else None,
            "median_spread": float(np.median(spreads)) if spreads else None,
            "mean_ask_size": float(np.mean(depths_ask)) if depths_ask else None,
            "mean_bid_size": float(np.mean(depths_bid)) if depths_bid else None,
            "n_with_spread": len(spreads),
            "n_with_depth": len(depths_ask),
        })
    return out


def Main() -> int:
    LowerToNormal()
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Confirm the frozen hash has not been altered.
    spec = json.loads((OUT_DIR / f"{BASE}_prereg_spec.json").read_text(encoding="utf-8"))
    chk = dict(spec)
    chk.pop("spec_hash", None)
    raw = json.dumps(chk, sort_keys=True, separators=(",", ":")).encode("utf-8")
    got = hashlib.sha1(raw).hexdigest()
    if got != PREREG_HASH or spec.get("spec_hash") != PREREG_HASH:
        raise SystemExit(f"prereg hash mismatch: file={spec.get('spec_hash')} recomputed={got} frozen={PREREG_HASH}")
    Log(f"prereg hash ok {PREREG_HASH} git={GitHash()}")

    WriteJson(OUT_DIR / "_status.json", {"phase": "fit"})
    players, games = LoadSeasons()
    feats, _, _ = BuildFeatures(players, games)
    train = [r for r in feats if r["season"] == TRAIN_SEASON and r.get("date") and r["date"] < EVAL_START]
    Log(f"features all={len(feats)} train={len(train)}")
    params_3pm = FitParams(train)
    AttachPredictions(train, params_3pm)
    AttachPredictions(feats, params_3pm)
    params_by = {"3pm": params_3pm}
    for fam in ("pts", "reb", "ast", "pra"):
        p = FitCountParams(train, fam)
        AttachCountPredictions(feats, p, fam)
        params_by[fam] = p
        Log(f"fit {fam} n={p['n_fit']} mu_rate={p['mu_rate']:.4f} k={p['k_rate']:.1f} alpha={p['alpha']:.3f}")
    WriteJson(OUT_DIR / f"{BASE}_params.json", {k: v for k, v in params_by.items()})

    by_name, by_last = BuildIndex(feats)
    WriteJson(OUT_DIR / "_status.json", {"phase": "score"})
    universe, coverage = LoadMarketsAndQuotes()
    WriteJson(OUT_DIR / f"{BASE}_coverage.json", coverage)
    Log("coverage " + json.dumps(coverage))
    n_quoted = sum(1 for r in universe if r.get("quoted"))
    if n_quoted == 0:
        # Blocker ? no T-60 quotes.
        lines = [
            "# Blocker: no T-60 Kalshi player-prop price history",
            "",
            "Written after the frozen preregistration "
            f"(spec hash `{PREREG_HASH}`). Simulation only. No wager was placed. No edge is claimed.",
            "",
            "## What was tried",
            "",
            "- Listed series via `GET /series` and direct `GET /series/{ticker}` for the preregistered tickers.",
            "- Listed events via `GET /events?series_ticker=...` (settled and unfiltered).",
            "- Pulled settled markets from `GET /historical/markets` and `GET /markets?status=settled`.",
            "- Historical cutoff from `GET /historical/cutoff`.",
            "- Candles: `GET /historical/markets/{ticker}/candlesticks` for markets settled before the cutoff, "
            "and `GET /series/{series}/markets/{ticker}/candlesticks` for later markets, with `period_interval=1` "
            "then `60`, in a window around tip. Rate limit about 2.5 req/s with exponential backoff on 429s. "
            "Cache under `data\\\\props\\\\`.",
            "",
            "## Settled-market coverage (evaluation window starts 2025-04-14)",
            "",
            "| series | listed markets | with result | in eval window | candle files | empty candles | T-60 quotes | date range |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for c in coverage:
            lines.append(
                f"| {c['series']} | {c.get('n_markets') or 0} | {c.get('n_with_result') or 0} | "
                f"{c.get('n_eval') or 0} | {c.get('n_candle_files') or 0} | {c.get('n_empty_candles') or 0} | "
                f"{c.get('n_quoted_t60') or 0} | {c.get('date_min') or 'NA'} .. {c.get('date_max') or 'NA'} |"
            )
        lines += [
            "",
            f"**Verdict: PASS.** The preregistered hypothesis is not confirmed because T-60 price history was not obtained. "
            f"Code `feature/props-backtest` at `{GitHash()}`, not pushed.",
        ]
        (OUT_DIR / f"{BASE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        WriteCsv(OUT_DIR / f"{BASE}.csv", [{
            "date": "2026-11-13", "sport": "nba", "series": "nba_player_props",
            "verdict": "PASS", "reason": "no_t60_quotes", "spec_hash": PREREG_HASH,
            "spa_p": None, "mean_edge": None, "n_bets": 0, "git": GitHash(),
        }])
        (OUT_DIR / "_DONE.txt").write_text("verdict=PASS\nreason=no_t60_quotes\n", encoding="utf-8")
        Log("DONE blocker no T-60 quotes")
        return 0

    scored, skipped = ScoreUniverse(universe, by_name, by_last, params_by)
    Log(f"scored={len(scored)} skipped={skipped}")
    bets, hold, gap, spa, rule_ids, D = SpaAndCi(scored)
    spreads = Spreads(scored)
    mc = MonteCarlo(bets, paths=10000, seed=7) if bets else None

    spa_p = (spa or {}).get("spa_p")
    white_p = (spa or {}).get("white_p")
    edge_above = bool(hold and hold.get("lo") is not None and hold["lo"] > 0)
    spa_sig = spa_p is not None and spa_p < 0.05
    verdict = "FAIL_HYPOTHESIS_CONFIRMED" if (spa_sig and edge_above) else "PASS"
    # Keep the folder-facing word PASS when the hypothesis is not confirmed,
    # and CONFIRM only if the frozen rule fires. The md explains this.
    if spa_sig and edge_above:
        verdict = "CONFIRM"
    else:
        verdict = "PASS"

    WriteCsv(OUT_DIR / f"{BASE}_coverage.csv", [
        {"date": "2026-11-13", "sport": "nba", **c} for c in coverage
    ])
    WriteCsv(OUT_DIR / f"{BASE}_spreads.csv", spreads)
    WriteCsv(OUT_DIR / f"{BASE}_holdout-bets.csv", [
        {k: b[k] for k in (
            "date", "series", "ticker", "player_name", "k", "strike_bucket", "side",
            "model_p", "fill", "mid", "edge", "result", "pnl", "implied_no_ask",
            "spread", "quote_age_s", "tip_src", "match", "fee_basis",
        ) if k in b} for b in bets
    ])
    WriteCsv(OUT_DIR / f"{BASE}_trials.csv", [
        {"rule_id": rid, "mean_daily_pnl": float(D[:, j].mean()) if D.size else None,
         "n_days": int(D.shape[0]) if D.size else 0}
        for j, rid in enumerate(rule_ids)
    ])
    mc_rows = []
    if mc:
        for strat, key in (("flat100", "flat100"), ("qkelly", "qkelly")):
            got = mc.get(key) or {}
            mc_rows.append({
                "date": "2026-11-13", "sport": "nba", "series": "nba_player_props_mc",
                "strategy": strat, "paths": 10000, "start": 10000,
                "n_bets": mc.get("n_bets"), "n_days": mc.get("n_days"),
                "p_ruin": got.get("p_ruin"), "p5": got.get("p5"),
                "p50": got.get("p50"), "p95": got.get("p95"), "mean": got.get("mean"),
            })
    WriteCsv(OUT_DIR / f"{BASE}_mc.csv", mc_rows)
    WriteJson(OUT_DIR / f"{BASE}_spa.json", {"spa": spa, "rule_ids": rule_ids})
    WriteJson(OUT_DIR / f"{BASE}_skipped.json", skipped)

    lines = [
        "# Kalshi NBA player-prop T-60 backtest ? 2026-11-13",
        "",
        f"Preregistered spec hash `{PREREG_HASH}` (written before scoring). "
        f"Simulation only. No wager was placed. Fee basis `{FEE_BASIS}`.",
        "",
        "## Coverage",
        "",
        "| series | listed | with result | eval window | candle files | T-60 quotes | date range |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for c in coverage:
        lines.append(
            f"| {c['series']} | {c.get('n_markets') or 0} | {c.get('n_with_result') or 0} | "
            f"{c.get('n_eval') or 0} | {c.get('n_candle_files') or 0} | {c.get('n_quoted_t60') or 0} | "
            f"{c.get('date_min') or 'NA'} .. {c.get('date_max') or 'NA'} |"
        )
    lines += [
        "",
        f"Scored markets (joined to a leak-free pregame model row): {len(scored)}. "
        f"Qualifying bets (edge >= 3c after the fee, one side per market): {len(bets)}. Skips: {skipped}.",
        "",
        "## Log-loss gap vs T-60 mid (bits)",
        "",
        (f"Mean gap {Fmt(gap['mean'])} bits, 90% day-block CI [{Fmt(gap['lo'])}, {Fmt(gap['hi'])}], "
         f"N={gap['n']}, days={gap['n_clusters']}. Positive means the model beats the mid."
         if gap and gap.get("mean") is not None else "Not available (no quoted mids)."),
        "",
        "## Holdout realized edge",
        "",
        (f"N={hold['n_bets']} bets on {hold['n_days']} days, mean P&L/contract {Fmt(hold['mean'])}, "
         f"95% day-block CI [{Fmt(hold['lo'])}, {Fmt(hold['hi'])}]."
         if hold and hold.get("mean") is not None else "No qualifying bets."),
        "",
        "## SPA / White RC (10,000 day-block resamples, strike x series x side)",
        "",
        (f"Hansen SPA p={Fmt(spa_p)}, White p={Fmt(white_p)}, rules={len(rule_ids)}, days={0 if D is None else D.shape[0]}."
         if spa else "SPA not computed (insufficient days or rules)."),
        "",
        "## $10k Monte Carlo (10,000 paths)",
        "",
    ]
    if mc:
        for strat, label in (("flat100", "flat100"), ("qkelly", "quarter-Kelly, 2% cap")):
            got = mc.get(strat) or {}
            lines.append(
                f"- {label}: mean {Fmt(got.get('mean'), 0)}, p50 {Fmt(got.get('p50'), 0)}, "
                f"p5 {Fmt(got.get('p5'), 0)}, p95 {Fmt(got.get('p95'), 0)}, p_ruin {Fmt(got.get('p_ruin'), 3)}."
            )
    else:
        lines.append("Not run (no qualifying bets).")
    lines += ["", "## Book spread and depth at T-60", ""]
    if spreads:
        lines += [
            "| series | N quoted | mean spread | median spread | mean ask size | mean bid size |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for s in spreads:
            lines.append(
                f"| {s['series']} | {s['n_quoted']} | {Fmt(s['mean_spread'])} | {Fmt(s['median_spread'])} | "
                f"{Fmt(s['mean_ask_size'])} | {Fmt(s['mean_bid_size'])} |"
            )
    else:
        lines.append("No quoted markets.")
    lines += [
        "",
        f"**Verdict: {verdict}.** "
        "PASS unless SPA p < 0.05 and the holdout realized-edge CI is entirely above 0 after the fee "
        "(the frozen preregistered rule).",
        "",
        f"Code `feature/props-backtest` at `{GitHash()}`, not pushed. Elapsed {time.time()-t0:.0f}s.",
    ]
    (OUT_DIR / f"{BASE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    WriteCsv(OUT_DIR / f"{BASE}.csv", [{
        "date": "2026-11-13", "sport": "nba", "series": "nba_player_props",
        "verdict": verdict, "spec_hash": PREREG_HASH,
        "n_scored": len(scored), "n_bets": len(bets),
        "mean_edge": None if not hold else hold.get("mean"),
        "edge_lo95": None if not hold else hold.get("lo"),
        "edge_hi95": None if not hold else hold.get("hi"),
        "gap_bits": None if not gap else gap.get("mean"),
        "gap_lo90": None if not gap else gap.get("lo"),
        "gap_hi90": None if not gap else gap.get("hi"),
        "spa_p": spa_p, "white_p": white_p, "n_rules": len(rule_ids),
        "fee_basis": FEE_BASIS, "git": GitHash(),
    }])
    (OUT_DIR / "_DONE.txt").write_text(f"verdict={verdict}\nhash={PREREG_HASH}\n", encoding="utf-8")
    WriteJson(OUT_DIR / "_status.json", {"phase": "done", "verdict": verdict, "elapsed_s": time.time() - t0})
    Log(f"DONE verdict={verdict} bets={len(bets)} spa_p={spa_p} elapsed={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
