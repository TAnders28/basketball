"""Build the in-game WP model, calibrate it, and replay it vs Kalshi 1-minute candles."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np

from InPlay.Align import AsOf
from InPlay.Load import LoadGames
from InPlay.WinProb import (
    AttachGbmToStates, AttachMcToStates, BuildMcGrid, EstimateRates, FEATURE_NAMES,
    LimitThreads, ListEspnPaths, LoadPriors, N_WORKERS, ParseAllEspn, PhaseOf,
    REG_END, SetNormalPriority, TimeBin, WalkForwardGbm, ClipP, LogLoss, Brier,
    _HomeAdvantagePpp,
)
from Backtest.ResidualSearch import DayBlockBootstrap, HansenSpa, StationaryBootstrapIndices

OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_inplay-wp")
BASE = "2026-11-13_inplay-wp"
GAMES_OBJ = os.path.join("data", "inplay", "games_obj_s0.pkl")  # no Polymarket shift
PARSE_CACHE = Path("data/inplay/wp_parsed.pkl")
FEE = 0.02
FEE_BASIS = "tim_est_0.02"
EDGE_CUTS = (0.03, 0.05, 0.08, 0.10, 0.12)
PHASES = ("all", "Q1", "Q2", "Q3", "Q4", "last5")
MODELS = ("gbm", "mc")
MC_SIMS = 1_000_000
MC_EDGES = {"e-4": -0.04, "e-2": -0.02, "e0": 0.0, "e2": 0.02, "e4": 0.04}
BINS = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]


def WriteJson(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def WriteCsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in fields:
                v = r.get(k, "")
                if v is None:
                    out[k] = ""
                elif isinstance(v, float):
                    out[k] = f"{v:.10g}"
                else:
                    out[k] = v
            w.writerow(out)


def GitHash():
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def PeakSampler(stop_flag, out_path, interval=3.0):
    peak, samples = 0.0, 0
    try:
        import subprocess
        while not stop_flag["stop"]:
            try:
                raw = subprocess.check_output(
                    ["powershell", "-NoProfile", "-Command",
                     "(Get-Counter '\\Process(python*)\\% Processor Time' -ErrorAction SilentlyContinue | "
                     "Measure-Object -Property CookedValue -Sum).Sum"],
                    text=True, timeout=20)
                # This is already "% of one core" summed; convert to machine % by / nproc roughly later.
                val = float(raw.strip().split()[-1])
                peak = max(peak, val)
                samples += 1
                out_path.write_text(json.dumps({"peak_python_core_pct": peak, "samples": samples,
                                                "nproc": os.cpu_count()}), encoding="utf-8")
            except Exception:
                pass
            time.sleep(interval)
    except Exception:
        pass


def Reliability(pairs):
    """pairs: (p, y). Returns bin rows."""
    rows = []
    for lo, hi in BINS:
        sel = [(p, y) for p, y in pairs if (lo <= p < hi) or (hi == 1 and p <= 1 and p >= lo)]
        if hi == 1.0:
            sel = [(p, y) for p, y in pairs if lo <= p <= hi]
        n = len(sel)
        rows.append({
            "bin_lo": lo, "bin_hi": hi, "n": n,
            "mean_pred": (sum(p for p, _ in sel) / n) if n else None,
            "obs_rate": (sum(y for _, y in sel) / n) if n else None,
        })
    return rows


def MetricsPairs(pairs):
    if not pairs:
        return {"n": 0, "log_loss": None, "brier": None, "ece": None}
    n = len(pairs)
    ll = sum(LogLoss(p, y) for p, y in pairs) / n
    br = sum(Brier(p, y) for p, y in pairs) / n
    ece = 0.0
    for row in Reliability(pairs):
        if row["n"]:
            ece += (row["n"] / n) * abs(row["mean_pred"] - row["obs_rate"])
    return {"n": n, "log_loss": ll, "brier": br, "ece": ece}


def AsOfState(states, t):
    ts = [s["t"] for s in states]
    i = 0
    # last state with t <= t
    import bisect
    j = bisect.bisect_right(ts, t) - 1
    return states[j] if j >= 0 else None


def BuildBarTable(parsed, kalshi_games):
    """1-minute in-play bars with model, ESPN WP, and Kalshi quotes (no time shift)."""
    by_id = {g["game_id"]: g for g in parsed}
    rows = []
    n_miss_parse = n_miss_model = 0
    for g in kalshi_games:
        rec = by_id.get(str(g.gid))
        if rec is None or not rec.get("states"):
            n_miss_parse += 1
            continue
        for i in range(g.n):
            if not g.el or g.el[i] < 0:
                continue
            # in-play only (already the usual filter in LoadGames, but keep tip/end sane)
            mid, ask, bid = g.mid[i], g.ask[i], g.bid[i]
            if mid is None or ask is None:
                continue
            st = AsOfState(rec["states"], g.t[i])
            if st is None:
                n_miss_model += 1
                continue
            p_gbm = st.get("p_gbm")
            p_mc = st.get("p_mc")
            wp = g.wp[i] if g.wp else st.get("wp_espn")
            rows.append({
                "game_id": str(g.gid), "date": g.day, "season": int(g.season),
                "t": int(g.t[i]), "el": float(g.el[i]), "per": int(g.per[i]),
                "phase": PhaseOf(g.el[i], g.per[i]),
                "won": int(g.won),
                "p_gbm": p_gbm, "p_mc": p_mc, "p_espn": wp,
                "mid": float(mid), "ask": float(ask), "bid": float(bid) if bid is not None else None,
                "prior": rec["prior"], "prior_source": rec["prior_source"],
                "pre": g.pre,
            })
    print(f"bars={len(rows)} miss_parse={n_miss_parse} miss_state={n_miss_model} games_in={len(kalshi_games)}",
          flush=True)
    return rows


def SplitTwoThirds(game_ids_sorted):
    n = len(game_ids_sorted)
    cut = int(math.floor(n * 2 / 3))
    tune, hold = set(game_ids_sorted[:cut]), set(game_ids_sorted[cut:])
    return tune, hold, cut


def Qualifies(p, mid, ask, bid, fee=FEE, cut=0.03):
    if p is None or ask is None or bid is None:
        return None
    e_h = float(p) - (float(ask) + fee)
    away_ask = 1.0 - float(bid)
    e_a = (1.0 - float(p)) - (away_ask + fee)
    if e_h >= e_a:
        side, fill, edge, y_side = "home", float(ask), e_h, None
        return side, fill, edge, "home"
    return "away", away_ask, e_a, "away"


def FirstSignals(bars, game_ids, model_key, thresh, phase, fee=FEE):
    """One trade per game: first qualifying bar in the allowed phase, hold to settlement."""
    by = defaultdict(list)
    for r in bars:
        if r["game_id"] not in game_ids:
            continue
        if phase != "all" and r["phase"] != phase:
            continue
        p = r.get(f"p_{model_key}") if model_key != "espn" else r.get("p_espn")
        if p is None or r["ask"] is None or r["bid"] is None:
            continue
        got = Qualifies(p, r["mid"], r["ask"], r["bid"], fee=fee, cut=thresh)
        if got is None:
            continue
        side, fill, edge, _ = got
        if edge < thresh:
            continue
        by[r["game_id"]].append((r, side, fill, edge, p))
    trades = []
    for gid, evs in by.items():
        evs.sort(key=lambda x: x[0]["t"])
        r, side, fill, edge, p = evs[0]
        y = int(r["won"]) if side == "home" else 1 - int(r["won"])
        pnl = y - (fill + fee)
        trades.append({
            "game_id": gid, "date": r["date"], "season": r["season"], "t": r["t"],
            "phase": r["phase"], "side": side, "model_prob": p, "fill": fill,
            "mid": r["mid"], "edge_claimed": edge, "result": y, "pnl": pnl,
            "prior_source": r["prior_source"],
        })
    return trades


def MeanPnlCi(trades):
    if not trades:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_games": 0, "wins": 0}
    pnls = [t["pnl"] for t in trades]
    clusters = [t["game_id"] for t in trades]
    ci = DayBlockBootstrap(pnls, clusters, reps=10000, seed=7, level=0.95)
    ci["n_games"] = len({t["game_id"] for t in trades})
    ci["wins"] = int(sum(t["result"] for t in trades))
    return ci


def RuleId(model, thresh, phase):
    return f"{model}_t{int(thresh*100):02d}_{phase}"


def BuildRules():
    return [{"model": m, "thresh": th, "phase": ph,
             "rule_id": RuleId(m, th, ph)}
            for m in MODELS for th in EDGE_CUTS for ph in PHASES]


def SpaFromRules(rule_trades, hold_ids, all_dates_by_game):
    """Game-level loss differentials vs 0 (mean pnl), one column per rule, rows = holdout games."""
    hold_games = sorted(hold_ids)
    # Use all holdout games (0 if no trade) so rules with fewer trades are comparable.
    # Actually SPA on days/games with at least one rule trading is OK; include all holdout games.
    K = []
    cols = []
    for rule_id, trades in rule_trades.items():
        by = {t["game_id"]: t["pnl"] for t in trades}
        col = np.array([by.get(gid, 0.0) for gid in hold_games], dtype=np.float64)
        cols.append(col)
        K.append(rule_id)
    if not cols:
        return None, K
    D = np.column_stack(cols)  # T x K, positive = profit (better than 0)
    return HansenSpa(D, reps=10000, seed=7, mean_block=5), K


def ScoringRunEvents(parsed, kalshi_by_id, run_pts=8):
    """t0 = play that extends a same-team scoring run to >= run_pts. As-of Kalshi mids, no future."""
    events = []
    for rec in parsed:
        g = kalshi_by_id.get(rec["game_id"])
        if g is None:
            continue
        times, mids = [], []
        for i in range(g.n):
            if g.mid[i] is None:
                continue
            times.append(g.t[i])
            mids.append(g.mid[i])
        if len(times) < 10:
            continue
        prev = 0
        for s in rec["states"]:
            run = int(s.get("run") or 0)
            # Fire when |run| first reaches run_pts (or jumps past it) on a scoring play.
            if s.get("scoring") and abs(run) >= run_pts and abs(prev) < run_pts:
                t0 = s["t"]
                direction = 1 if run > 0 else -1  # home mid should rise if direction=1
                m0 = AsOf(times, mids, t0)  # last mid at or before t0 (no future)
                m1 = AsOf(times, mids, t0 + 60)
                m2 = AsOf(times, mids, t0 + 120)
                m5 = AsOf(times, mids, t0 + 300)
                if m0 is None or m5 is None:
                    prev = run
                    continue
                move5 = (m5 - m0) * direction
                ev = {
                    "game_id": rec["game_id"], "t0": t0, "run": run,
                    "m0": m0, "m1": m1, "m2": m2, "m5": m5,
                    "move_1": None if m1 is None else (m1 - m0) * direction,
                    "move_2": None if m2 is None else (m2 - m0) * direction,
                    "move_5": move5,
                    "frac_1": None if m1 is None or abs(move5) < 1e-9 else ((m1 - m0) * direction) / move5,
                    "frac_2": None if m2 is None or abs(move5) < 1e-9 else ((m2 - m0) * direction) / move5,
                    "remain_after_1": None if m1 is None else (m5 - m1) * direction,
                }
                events.append(ev)
            prev = run
    return events


def Fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def FmtCi(d, nd=4):
    if not d or d.get("mean") is None:
        return "NA"
    return f"{Fmt(d['mean'], nd)} [{Fmt(d['lo'], nd)}, {Fmt(d['hi'], nd)}]"


def WriteMd(path, cal, best, hold_ci, spa, reprice, n_parse, n_bars, n_tune, n_hold,
            n_sims, hash_spec, git, elapsed, peak, skipped):
    spa_p = spa.get("spa_p") if spa else None
    edge_above = hold_ci and hold_ci.get("lo") is not None and hold_ci["lo"] > 0
    spa_sig = spa_p is not None and spa_p < 0.05
    verdict = "PASS"
    if spa_sig and edge_above:
        verdict = "BEATS MARKET"
    lines = []
    lines.append("# NBA in-play win-probability vs Kalshi (2026-11-13)")
    lines.append("")
    lines.append("Simulation only. No wagers were placed. Fee is Tim's $0.02 working estimate (`fee_basis=tim_est_0.02`).")
    lines.append("Kalshi candles are used with no time shift. Polymarket is not shifted forward (it lags about one minute).")
    lines.append("")
    lines.append("## Model")
    lines.append(f"- GBM (LightGBM, market logit as init-score) trained game-grouped walk-forward by season on ESPN PBP ({n_parse} games). Features: margin, time left, possession, fouls/bonus, timeouts, pregame prior, scoring run.")
    lines.append("- Pregame prior: Kalshi T-60 de-vig mid, else stack_global, else a de-vigged book/pickcenter ML, each flagged. A last-resort ESPN opening WP is used only for training coverage and is flagged `espn_wp`.")
    lines.append(f"- Possession MC grid: {n_sims:,} vectorized remaining-game sims per (time bin, possession, bonus, prior-edge) cell. Timeouts are in the GBM only, not the MC grid.")
    if skipped:
        for s in skipped:
            lines.append(f"- {s}")
    lines.append(f"- Frozen selected rule hash `{hash_spec}`. Code `feature/inplay-wp` at `{git}`, not pushed.")
    lines.append("")
    lines.append("## Calibration (same timestamps, in-play Kalshi bars)")
    lines.append("| model | phase | N | log loss | Brier | vs Kalshi dBrier |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in cal:
        lines.append(
            f"| {row['model']} | {row['phase']} | {row['n']} | {Fmt(row['log_loss'])} | {Fmt(row['brier'])} | {Fmt(row.get('dbrier_vs_kalshi'))} |"
        )
    lines.append("")
    lines.append("## Holdout replay (first signal, hold to settlement, ask + $0.02, 3c+ and variants)")
    if best:
        lines.append(f"- Selected on the first 2/3 of Kalshi-priced games (N_tune_games={n_tune}): `{best['rule_id']}`.")
        lines.append(f"- Holdout (last 1/3, N_games={n_hold}): N_bets={hold_ci.get('n', 0)}, mean realized P&L/contract {FmtCi(hold_ci)}.")
        lines.append(f"- SPA p (Hansen consistent, all threshold x phase x model rules, 10,000 game-block resamples) = {Fmt(spa_p)}.")
    else:
        lines.append("- No qualifying trades on the tune split.")
    lines.append("")
    lines.append("## Reprice after scoring runs (8+ unanswered, Kalshi mid, no shift)")
    if reprice:
        lines.append(
            f"- N={reprice['n']}: mean move at +1 / +2 / +5 min = "
            f"{Fmt(reprice['m1'], 4)} / {Fmt(reprice['m2'], 4)} / {Fmt(reprice['m5'], 4)}."
        )
        lines.append(
            f"- Fraction of the +5-min move already in by +1 min: {Fmt(reprice['frac1'], 3)}; "
            f"remaining after +1 min: {Fmt(reprice['remain1'], 4)}."
        )
        lines.append("- Last night's play-level study found Kalshi reprices within 5-10 seconds on made field goals. This minute-level run check is consistent with no usable lag beyond 1 minute." if (reprice.get("remain1") is None or abs(reprice.get("remain1") or 0) < 0.02) else "- Some move remains after 1 minute; see the reprice CSV.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("- ESPN timestamps are not a live feed; last night showed most of a made-FG move is done in 5-10 seconds, which a one-minute bar cannot capture.")
    lines.append("- Candles have no size; fills at the ask are optimistic.")
    lines.append("- Away ask is implied as 1 - home bid.")
    lines.append("- The selected rule is chosen on the tune games; SPA is the multiple-testing check on the holdout.")
    if peak:
        lines.append(f"- Peak sampler: {peak}. Elapsed {elapsed:.0f}s. Workers={N_WORKERS}.")
    lines.append("")
    lines.append(f"**Verdict: {verdict}.** PASS unless SPA p < 0.05 and the holdout realized-edge CI is entirely above 0 after the fee.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return verdict


def Main() -> int:
    LimitThreads()
    SetNormalPriority()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = OUT_DIR / "_status.json"
    cpu_path = OUT_DIR / "_cpu_peak.json"
    import threading
    stop = {"stop": False}
    threading.Thread(target=PeakSampler, args=(stop, cpu_path), daemon=True).start()
    t0 = time.time()
    print(f"python={sys.version.split()[0]} workers={N_WORKERS} git={GitHash()}", flush=True)
    WriteJson(status, {"phase": "parse"})

    skipped = []
    priors = LoadPriors()
    print(f"prior keys={len(priors)}", flush=True)

    if PARSE_CACHE.exists():
        import pickle
        print(f"loading parse cache {PARSE_CACHE}", flush=True)
        parsed = pickle.loads(PARSE_CACHE.read_bytes())
    else:
        parsed = ParseAllEspn(workers=N_WORKERS)
        import pickle
        PARSE_CACHE.write_bytes(pickle.dumps(parsed, protocol=pickle.HIGHEST_PROTOCOL))
        print(f"cached {len(parsed)} games -> {PARSE_CACHE}", flush=True)
    src_counts = defaultdict(int)
    for g in parsed:
        src_counts[g["prior_source"]] += 1
    print(f"parsed={len(parsed)} prior_sources={dict(src_counts)}", flush=True)
    WriteJson(status, {"phase": "gbm", "n_parsed": len(parsed)})

    models = WalkForwardGbm(parsed)
    AttachGbmToStates(parsed, models)
    WriteJson(status, {"phase": "mc"})

    # Rates from pre-Kalshi / pre-holdout seasons only (ending year <= 2025, i.e. through 2024-25).
    rate_games = [g for g in parsed if (g.get("season") or 0) <= 2024]
    rates = EstimateRates(rate_games or parsed)
    print("rates", rates, flush=True)
    n_sims = MC_SIMS
    try:
        grid = BuildMcGrid(rates, n_sims, MC_EDGES, workers=N_WORKERS)
    except MemoryError:
        n_sims = 250_000
        skipped.append(f"MC reduced to {n_sims:,} sims/cell after a memory error (1M was requested).")
        grid = BuildMcGrid(rates, n_sims, MC_EDGES, workers=N_WORKERS)
    AttachMcToStates(parsed, grid, rates, MC_EDGES)

    WriteJson(status, {"phase": "bars"})
    G = LoadGames(GAMES_OBJ)
    kalshi = G.get("kalshi") or []
    print(f"kalshi games loaded={len(kalshi)} from {GAMES_OBJ}", flush=True)
    bars = BuildBarTable(parsed, kalshi)
    kalshi_by_id = {str(g.gid): g for g in kalshi}

    # Calibration vs Kalshi and ESPN on the same bars, by phase.
    cal_rows = []
    phases = ("all", "Q1", "Q2", "Q3", "Q4", "last5", "OT")
    for model, pkey in (("gbm", "p_gbm"), ("mc", "p_mc"), ("espn", "p_espn"), ("kalshi", "mid")):
        for phase in phases:
            sub = [b for b in bars if (phase == "all" or b["phase"] == phase) and b.get(pkey) is not None]
            pairs = [(float(b[pkey]), int(b["won"])) for b in sub]
            met = MetricsPairs(pairs)
            k_pairs = [(float(b["mid"]), int(b["won"])) for b in sub if b.get("mid") is not None]
            kmet = MetricsPairs(k_pairs)
            cal_rows.append({
                "date": "2026-11-13", "sport": "nba", "series": f"nba_inplay_{model}",
                "model": model, "phase": phase, "n": met["n"],
                "log_loss": met["log_loss"], "brier": met["brier"], "ece": met["ece"],
                "kalshi_log_loss": kmet["log_loss"], "kalshi_brier": kmet["brier"],
                "dbrier_vs_kalshi": (None if met["brier"] is None or kmet["brier"] is None
                                     else met["brier"] - kmet["brier"]),
                "dll_vs_kalshi": (None if met["log_loss"] is None or kmet["log_loss"] is None
                                  else met["log_loss"] - kmet["log_loss"]),
            })
    rel_rows = []
    for model, pkey in (("gbm", "p_gbm"), ("mc", "p_mc"), ("espn", "p_espn"), ("kalshi", "mid")):
        for phase in phases:
            sub = [b for b in bars if (phase == "all" or b["phase"] == phase) and b.get(pkey) is not None]
            for row in Reliability([(float(b[pkey]), int(b["won"])) for b in sub]):
                rel_rows.append({
                    "date": "2026-11-13", "sport": "nba", "series": f"nba_inplay_{model}",
                    "model": model, "phase": phase, **row,
                })

    # Replay: first 2/3 / last 1/3 of unique Kalshi games in the bar table, chronological.
    gids = []
    seen = set()
    for b in sorted(bars, key=lambda r: (r["date"], r["game_id"])):
        if b["game_id"] not in seen:
            seen.add(b["game_id"])
            gids.append(b["game_id"])
    tune_ids, hold_ids, cut = SplitTwoThirds(gids)
    print(f"replay games={len(gids)} tune={len(tune_ids)} hold={len(hold_ids)}", flush=True)

    rules = BuildRules()
    trial_rows = []
    rule_hold_trades = {}
    best, best_s = None, -9e9
    for rule in rules:
        tr_tune = FirstSignals(bars, tune_ids, rule["model"], rule["thresh"], rule["phase"])
        tr_hold = FirstSignals(bars, hold_ids, rule["model"], rule["thresh"], rule["phase"])
        ci_t, ci_h = MeanPnlCi(tr_tune), MeanPnlCi(tr_hold)
        rule_hold_trades[rule["rule_id"]] = tr_hold
        rec = {
            **rule, "n_tune": ci_t["n"], "tune_mean": ci_t["mean"], "tune_lo": ci_t["lo"], "tune_hi": ci_t["hi"],
            "n_hold": ci_h["n"], "hold_mean": ci_h["mean"], "hold_lo": ci_h["lo"], "hold_hi": ci_h["hi"],
            "fee_basis": FEE_BASIS,
        }
        trial_rows.append(rec)
        # Select on tune: require at least 30 trades, rank by mean realized pnl.
        if ci_t["n"] and ci_t["n"] >= 30 and ci_t["mean"] is not None and ci_t["mean"] > best_s:
            best, best_s = rec, ci_t["mean"]
    if best is None and trial_rows:
        # fallback: most trades on tune
        best = max(trial_rows, key=lambda r: (r["n_tune"] or 0, r["tune_mean"] or -9))

    spec = dict(rule_id=best["rule_id"] if best else None, model=best["model"] if best else None,
                thresh=best["thresh"] if best else None, phase=best["phase"] if best else None,
                fee=FEE, fee_basis=FEE_BASIS, entry="first_signal_hold_to_end",
                tune_fraction=2 / 3, n_tune_games=len(tune_ids), n_hold_games=len(hold_ids),
                features=FEATURE_NAMES, mc_sims=n_sims, games_obj=GAMES_OBJ,
                edge_cuts=list(EDGE_CUTS), phases=list(PHASES), models=list(MODELS))
    spec_hash = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    spec["spec_hash"] = spec_hash
    WriteJson(OUT_DIR / f"{BASE}_spec.json", spec)
    print(f"selected {spec.get('rule_id')} hash={spec_hash}", flush=True)

    spa, spa_ids = SpaFromRules(rule_hold_trades, hold_ids, None)
    if spa:
        print(f"SPA spa_p={spa.get('spa_p')} white_p={spa.get('white_p')}", flush=True)

    hold_ci = MeanPnlCi(rule_hold_trades.get(best["rule_id"], [])) if best else {"mean": None, "n": 0}

    # Reprice
    evs = ScoringRunEvents(parsed, kalshi_by_id, run_pts=8)
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None
    reprice = {
        "n": len(evs),
        "m1": _mean([e["move_1"] for e in evs]),
        "m2": _mean([e["move_2"] for e in evs]),
        "m5": _mean([e["move_5"] for e in evs]),
        "frac1": _mean([e["frac_1"] for e in evs if e["frac_1"] is not None and e["move_5"] and e["move_5"] > 0]),
        "frac2": _mean([e["frac_2"] for e in evs if e["frac_2"] is not None and e["move_5"] and e["move_5"] > 0]),
        "remain1": _mean([e["remain_after_1"] for e in evs]),
    }
    print("reprice", reprice, flush=True)

    # Outputs
    WriteCsv(OUT_DIR / f"{BASE}_calibration.csv", cal_rows,
             ["date", "sport", "series", "model", "phase", "n", "log_loss", "brier", "ece",
              "kalshi_log_loss", "kalshi_brier", "dbrier_vs_kalshi", "dll_vs_kalshi"])
    WriteCsv(OUT_DIR / f"{BASE}_reliability.csv", rel_rows,
             ["date", "sport", "series", "model", "phase", "bin_lo", "bin_hi", "n", "mean_pred", "obs_rate"])
    WriteCsv(OUT_DIR / f"{BASE}_trials.csv", trial_rows,
             ["rule_id", "model", "thresh", "phase", "n_tune", "tune_mean", "tune_lo", "tune_hi",
              "n_hold", "hold_mean", "hold_lo", "hold_hi", "fee_basis"])
    reprice_rows = [{
        "date": "2026-11-13", "sport": "nba", "series": "nba_inplay_reprice",
        "run_pts": 8, "n": reprice["n"], "move_1": reprice["m1"], "move_2": reprice["m2"],
        "move_5": reprice["m5"], "frac_1": reprice["frac1"], "frac_2": reprice["frac2"],
        "remain_after_1": reprice["remain1"],
    }]
    WriteCsv(OUT_DIR / f"{BASE}_reprice.csv", reprice_rows,
             ["date", "sport", "series", "run_pts", "n", "move_1", "move_2", "move_5",
              "frac_1", "frac_2", "remain_after_1"])
    # Holdout trades of the selected rule, tidy
    sel_trades = rule_hold_trades.get(best["rule_id"], []) if best else []
    WriteCsv(OUT_DIR / f"{BASE}_holdout-trades.csv", sel_trades,
             ["game_id", "date", "season", "t", "phase", "side", "model_prob", "fill", "mid",
              "edge_claimed", "result", "pnl", "prior_source"])

    peak = None
    if cpu_path.exists():
        try:
            peak = json.loads(cpu_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    verdict = WriteMd(
        OUT_DIR / f"{BASE}.md", cal_rows, best, hold_ci, spa, reprice,
        len(parsed), len(bars), len(tune_ids), len(hold_ids), n_sims, spec_hash,
        GitHash(), time.time() - t0, peak, skipped,
    )
    summary = [{
        "date": "2026-11-13", "sport": "nba", "series": "nba_inplay_wp",
        "model": best["model"] if best else "", "rule_id": best["rule_id"] if best else "",
        "spec_hash": spec_hash, "n_parsed": len(parsed), "n_bars": len(bars),
        "n_tune_games": len(tune_ids), "n_hold_games": len(hold_ids),
        "n_bets": hold_ci.get("n"), "mean_edge": hold_ci.get("mean"),
        "edge_lo95": hold_ci.get("lo"), "edge_hi95": hold_ci.get("hi"),
        "spa_p": spa.get("spa_p") if spa else None,
        "white_p": spa.get("white_p") if spa else None,
        "reprice_n": reprice["n"], "reprice_frac1": reprice["frac1"],
        "verdict": verdict, "fee_basis": FEE_BASIS, "mc_sims": n_sims,
    }]
    WriteCsv(OUT_DIR / f"{BASE}.csv", summary,
             ["date", "sport", "series", "model", "rule_id", "spec_hash", "n_parsed", "n_bars",
              "n_tune_games", "n_hold_games", "n_bets", "mean_edge", "edge_lo95", "edge_hi95",
              "spa_p", "white_p", "reprice_n", "reprice_frac1", "verdict", "fee_basis", "mc_sims"])
    stop["stop"] = True
    WriteJson(status, {"phase": "done", "verdict": verdict, "rule": spec.get("rule_id"),
                       "spa_p": spa.get("spa_p") if spa else None, "elapsed_s": time.time() - t0,
                       "peak": peak})
    (OUT_DIR / "_DONE.txt").write_text(f"verdict={verdict}\nrule={spec.get('rule_id')}\n", encoding="utf-8")
    print(f"DONE verdict={verdict} elapsed={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
