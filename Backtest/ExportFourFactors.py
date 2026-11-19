"""Score the four-factor GAP prototype on the 2026-11-13 calibration game set."""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

MAIN = Path(r"E:\AgentWorkspaces\Basketball\basketball")
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-13_four-factors")
BASE = "2026-11-13_four-factors"
SEASON_DIR = MAIN / "data" / "season"
EDGES_IN = Path(r"E:\AgentDocs\Basketbot\2026-11-13_bankroll-10k\2026-11-13_bankroll-10k_edges.csv")
BURNIN_END = "2025-12-02"


def LimitCpu() -> None:
    """Normal priority, at most ~40% of cores via thread caps (this job is single-process)."""
    n = os.cpu_count() or 8
    cap = max(1, int(n * 0.40))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(min(cap, 1))  # one thread in this process; 40% would apply if we pooled
    # Stay at Normal (do not LowerPriority). Record the cap for the log.
    os.environ["BASKETBOT_MAX_WORKERS"] = str(cap)


def GitHash(repo: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(repo), text=True).strip()


def LoadTeamGames(season: int) -> dict:
    path = SEASON_DIR / f"team_games_{season}.jsonl"
    by = defaultdict(dict)
    if not path.exists():
        return by
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            gid = str(r["GAME_ID"])
            team = r["TEAM_ABBREVIATION"]
            by[gid][team] = r
    return by


def LoadSeasonGames(season: int) -> list:
    path = SEASON_DIR / f"games_{season}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def BuildWalkGames(seasons) -> list:
    out = []
    missing_box = 0
    for s in seasons:
        boxes = LoadTeamGames(s)
        for g in LoadSeasonGames(s):
            if g.get("home_score") is None:
                continue
            gid = str(g["game_id"])
            sides = boxes.get(gid, {})
            home, away = g["home_team"], g["away_team"]
            if home not in sides or away not in sides:
                missing_box += 1
                continue
            out.append({
                "game_id": gid, "game_date": g["game_date"], "season": s,
                "home": home, "away": away,
                "home_box": sides[home], "away_box": sides[away],
            })
    out.sort(key=lambda g: (g["game_date"], g["game_id"]))
    return out, missing_box


def FitLogitX(cols, y, l2=1e-6):
    import numpy as np
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 30:
        return None
    X = np.column_stack([np.ones(n)] + [np.asarray(c, dtype=float) for c in cols])
    k = X.shape[1]
    theta = np.zeros(k)
    # Start with the last column (usually the market) at weight 1.
    if k >= 2:
        theta[-1] = 1.0
    eye = np.eye(k)
    for _ in range(40):
        z = np.clip(X @ theta, -30.0, 30.0)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-6, 1.0 - 1e-6)
        grad = X.T @ (p - y) + l2 * theta
        w = p * (1.0 - p)
        H = X.T @ (w[:, None] * X) + l2 * eye
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, grad, rcond=None)[0]
        theta = theta - step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return {"theta": [float(x) for x in theta], "n": n}


def Sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def Logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def WeekStart(d):
    from datetime import date, timedelta
    dt = date.fromisoformat(d)
    return (dt - timedelta(days=dt.weekday())).isoformat()


def WalkBlendCols(games, col_names, burnin_end=BURNIN_END):
    """col_names are keys on each game, e.g. ['p_ff','p_mkt'] or ['p_stack','p_ff','p_mkt']."""
    from collections import defaultdict
    by_week = defaultdict(list)
    for g in games:
        by_week[WeekStart(g["game_date"])].append(g)
    weights = []
    pred_key = "p_blend_" + "_".join(col_names)
    for wk in sorted(by_week):
        prior = [g for g in games if g["game_date"] < wk and all(g.get(c) is not None for c in col_names)]
        fit = None
        if wk >= burnin_end and prior:
            fit = FitLogitX([[Logit(g[c]) for g in prior] for c in col_names],
                            [g["home_win"] for g in prior])
        row = {"week_start": wk, "n_prior": len(prior), "n_week": len(by_week[wk]), "fit": fit}
        for i, c in enumerate(col_names):
            row[f"w_{c}"] = None if not fit else fit["theta"][i + 1]
        row["a"] = None if not fit else fit["theta"][0]
        for g in by_week[wk]:
            if not fit or any(g.get(c) is None for c in col_names):
                g[pred_key] = None
            else:
                z = fit["theta"][0] + sum(fit["theta"][i + 1] * Logit(g[c]) for i, c in enumerate(col_names))
                g[pred_key] = Sigmoid(z)
        weights.append(row)
    return pred_key, weights


def BootstrapTheta(games, col_names, burnin_end=BURNIN_END, reps=2000, seed=7):
    import random
    from collections import defaultdict
    last = max((g["game_date"] for g in games if g.get("p_blend_" + "_".join(col_names)) is not None), default=None)
    if last is None:
        return {}
    wk = WeekStart(last)
    train = [g for g in games if g["game_date"] < wk and all(g.get(c) is not None for c in col_names)]
    point = FitLogitX([[Logit(g[c]) for g in train] for c in col_names], [g["home_win"] for g in train])
    if not point:
        return {}
    bucket = defaultdict(list)
    for g in train:
        bucket[g["game_date"]].append(g)
    keys = list(bucket)
    rng = random.Random(seed)
    boots = {i: [] for i in range(len(point["theta"]))}
    for _ in range(reps):
        draw = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        samp = []
        for k in draw:
            samp.extend(bucket[k])
        fit = FitLogitX([[Logit(g[c]) for g in samp] for c in col_names], [g["home_win"] for g in samp])
        if fit:
            for i, v in enumerate(fit["theta"]):
                boots[i].append(v)
    out = {"n": len(train), "n_days": len(keys), "a": point["theta"][0]}
    for i, c in enumerate(col_names):
        xs = sorted(boots[i + 1])
        out[c] = {
            "w": point["theta"][i + 1],
            "lo": xs[int(0.05 * len(xs))] if xs else None,
            "hi": xs[min(len(xs) - 1, int(0.95 * len(xs)))] if xs else None,
        }
    if boots[0]:
        xs = sorted(boots[0])
        out["a_lo"] = xs[int(0.05 * len(xs))]
        out["a_hi"] = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
    return out


def Main() -> int:
    t0 = time.time()
    os.chdir(MAIN)
    LimitCpu()
    sys.path.insert(0, str(MAIN))
    from Backtest.FourFactorsGap import WalkRatings
    from Backtest.ExportMarketBlend import (
        LoadPriced, AttachElo, PreparePath, GapBits, Metrics, WriteCsv, Fmt, LogLossNats, IgnBits,
        WORKTREE,
    )

    # Elo attach uses the worktree; do it via PreparePath then restore main on sys.path.
    PreparePath()
    priced = LoadPriced()
    AttachElo(priced)
    # FourFactorsGap lives in the main checkout; put MAIN first again after Elo.
    while str(MAIN) in sys.path:
        sys.path.remove(str(MAIN))
    sys.path.insert(0, str(MAIN))
    # Drop cached Backtest so FourFactorsGap loads from MAIN (worktree has no such module).
    for name in list(sys.modules):
        if name == "Backtest" or name.startswith("Backtest."):
            del sys.modules[name]
    from Backtest.FourFactorsGap import WalkRatings  # reimport from MAIN
    from Backtest.ExportMarketBlend import (LoadPriced as _LP, GapBits, Metrics, WriteCsv, Fmt,
                                            LogLossNats, IgnBits)

    print(f"priced {len(priced)}", flush=True)
    walk, missing_box = BuildWalkGames([2023, 2024, 2025, 2026])
    print(f"walk universe {len(walk)} missing_box={missing_box}", flush=True)
    preds = WalkRatings(walk)
    print(f"ratings for {len(preds)} games", flush=True)

    n_ff = 0
    for g in priced:
        pr = preds.get(g["game_id"])
        if pr is None:
            g["p_ff"] = None
        else:
            g["p_ff"] = float(pr["p_home"])
            n_ff += 1
    print(f"priced with four-factor p: {n_ff}", flush=True)
    have = [g for g in priced if g["p_ff"] is not None]
    if len(have) != len(priced):
        print(f"WARNING missing ff on {len(priced)-len(have)} priced games", flush=True)

    splits = {
        "holdout": [g for g in have if g["split"] == "holdout"],
        "forward": [g for g in have if g["split"] == "forward"],
        "pooled_oos": list(have),
    }
    gaps = {}
    for model, field in (("four_factors_gap", "p_ff"), ("stack_global", "p_stack"), ("elo_predictday_v1", "p_elo")):
        for split, gs in splits.items():
            gaps[(model, split)] = GapBits(gs, field)
            g = gaps[(model, split)]
            print(f"ign {model} {split}: {g['ign_rel_bits']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] n={g['n']}", flush=True)

    # Blends: ff+kalshi, and stack+ff+kalshi (four-factors as an input alongside stack_global).
    k2, w2 = WalkBlendCols(have, ["p_ff", "p_mkt"])
    k3, w3 = WalkBlendCols(have, ["p_stack", "p_ff", "p_mkt"])
    scored2 = [g for g in have if g.get(k2) is not None]
    scored3 = [g for g in have if g.get(k3) is not None]
    print(f"blend ff+mkt scored {len(scored2)}; stack+ff+mkt scored {len(scored3)}", flush=True)
    th2 = BootstrapTheta(have, ["p_ff", "p_mkt"])
    th3 = BootstrapTheta(have, ["p_stack", "p_ff", "p_mkt"])
    print("ff+mkt w_ff", th2.get("p_ff"), "w_mkt", th2.get("p_mkt"), flush=True)
    print("3way w_stack", th3.get("p_stack"), "w_ff", th3.get("p_ff"), "w_mkt", th3.get("p_mkt"), flush=True)

    for label, key, gs in (
        ("four_factors_kalshi_blend", k2, scored2),
        ("stack_ff_kalshi_blend", k3, scored3),
    ):
        for split, subset in (("holdout", [g for g in gs if g["split"] == "holdout"]),
                              ("forward", [g for g in gs if g["split"] == "forward"]),
                              ("pooled_oos", gs)):
            if subset:
                # temporarily point GapBits at the blend field
                tmp = []
                for g in subset:
                    h = dict(g)
                    h["p_blend_tmp"] = g[key]
                    tmp.append(h)
                # monkey: use a tiny wrapper
                from Backtest.ExportMarketBlend import IgnBits as IB, BootstrapMean
                ign_m = [IB(g[key], g["home_win"]) for g in subset]
                ign_k = [IB(g["p_mkt"], g["home_win"]) for g in subset]
                diffs = [a - b for a, b in zip(ign_m, ign_k)]
                days = [g["game_date"] for g in subset]
                game_ci = BootstrapMean(diffs, None)
                day_ci = BootstrapMean(diffs, days)
                mean = game_ci["mean"]
                gaps[(label, split)] = {
                    "ign_rel_bits": mean,
                    "lo": min(game_ci["lo"], day_ci["lo"]),
                    "hi": max(game_ci["hi"], day_ci["hi"]),
                    "lo_game": game_ci["lo"], "hi_game": game_ci["hi"],
                    "lo_day": day_ci["lo"], "hi_day": day_ci["hi"],
                    "n": len(subset), "n_days": day_ci["n_clusters"],
                    "growth_per_event": 2.0 ** (-mean) if mean is not None else None,
                    "ign_model": sum(ign_m) / len(subset),
                    "ign_market": sum(ign_k) / len(subset),
                }

    hash_main = GitHash(MAIN)
    hash_wt = GitHash(WORKTREE)
    asof = {}
    for g in have:
        asof[g["season_label"]] = g["game_date"]
    asof["pooled_oos"] = max(g["game_date"] for g in have)

    def add_summary(rows, model, field, games, season, split):
        pairs = [(g[field], g["home_win"]) for g in games if g.get(field) is not None]
        if not pairs:
            return
        last = max(g["game_date"] for g in games if g.get(field) is not None)
        met = Metrics(pairs)
        rows.append({
            "date": last, "sport": "nba", "series": f"nba_{model}", "model": model,
            "season": season, "split": split, **met,
        })

    summary = []
    for model, field, pool in (
        ("four_factors_gap", "p_ff", have),
        ("stack_global", "p_stack", have),
        ("market_kalshi_T60", "p_mkt", have),
        ("four_factors_kalshi_blend", k2, scored2),
        ("stack_ff_kalshi_blend", k3, scored3),
    ):
        add_summary(summary, model, field, [g for g in pool if g["split"] == "holdout"], "2025-26", "holdout")
        add_summary(summary, model, field, [g for g in pool if g["split"] == "forward"], "2026-27", "forward")
        add_summary(summary, model, field, pool, "pooled_oos", "pooled_oos")

    # Per-game csv for the priced set
    game_rows = []
    for g in have:
        game_rows.append({
            "date": g["game_date"], "sport": "nba", "season": g["season_label"], "split": g["split"],
            "event_id": g["game_id"], "event": g["event"],
            "home_win": g["home_win"],
            "p_four_factors_gap": g["p_ff"],
            "p_stack_global": g["p_stack"],
            "p_elo_predictday_v1": g["p_elo"],
            "p_market_kalshi_T60": g["p_mkt"],
            "p_ff_kalshi_blend": g.get(k2),
            "p_stack_ff_kalshi_blend": g.get(k3),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sum_fields = ["date", "sport", "series", "model", "season", "split",
                  "n", "accuracy", "brier", "log_loss", "ece"]
    game_fields = ["date","sport","season","split","event_id","event","home_win",
                   "p_four_factors_gap","p_stack_global","p_elo_predictday_v1","p_market_kalshi_T60",
                   "p_ff_kalshi_blend","p_stack_ff_kalshi_blend"]
    WriteCsv(OUT_DIR / f"{BASE}_summary.csv", summary, sum_fields)
    WriteCsv(OUT_DIR / f"{BASE}.csv", game_rows, game_fields)
    ign_rows = []
    for (model, split), g in gaps.items():
        ign_rows.append({
            "date": asof.get("2025-26" if split == "holdout" else ("2026-27" if split == "forward" else "pooled_oos"), asof["pooled_oos"]),
            "sport": "nba", "model": model, "split": split,
            "n": g["n"], "n_days": g["n_days"],
            "ign_rel_bits": g["ign_rel_bits"], "ci90_lo": g["lo"], "ci90_hi": g["hi"],
            "ci90_lo_game": g["lo_game"], "ci90_hi_game": g["hi_game"],
            "ci90_lo_day": g["lo_day"], "ci90_hi_day": g["hi_day"],
            "growth_per_event": g["growth_per_event"],
            "ign_model_bits": g["ign_model"], "ign_market_bits": g["ign_market"],
        })
    ign_fields = ["date","sport","model","split","n","n_days","ign_rel_bits","ci90_lo","ci90_hi",
                  "ci90_lo_game","ci90_hi_game","ci90_lo_day","ci90_hi_day","growth_per_event",
                  "ign_model_bits","ign_market_bits"]
    WriteCsv(OUT_DIR / f"{BASE}_ign.csv", ign_rows, ign_fields)

    # weights over time for both blends
    wrows = []
    for kind, weights in (("ff_mkt", w2), ("stack_ff_mkt", w3)):
        for w in weights:
            rec = {"blend": kind, "week_start": w["week_start"], "n_prior": w["n_prior"], "n_week": w["n_week"], "a": w["a"]}
            for k in w:
                if k.startswith("w_"):
                    rec[k] = w[k]
            wrows.append(rec)
    wfields = ["blend","week_start","n_prior","n_week","a","w_p_ff","w_p_mkt","w_p_stack"]
    WriteCsv(OUT_DIR / f"{BASE}_weights.csv", wrows, wfields)

    def ll(gs, field):
        xs = [g for g in gs if g.get(field) is not None]
        if not xs:
            return None, None, 0
        return (sum(LogLossNats(g[field], g["home_win"]) for g in xs) / len(xs),
                sum(IgnBits(g[field], g["home_win"]) for g in xs) / len(xs), len(xs))

    # Verdict on four-factors as a stack input: w_ff CI includes 0?
    wff = (th3.get("p_ff") or {})
    includes_zero = wff.get("lo") is not None and wff.get("hi") is not None and wff["lo"] <= 0.0 <= wff["hi"]
    near_zero = wff.get("w") is not None and abs(wff["w"]) < 0.05
    pass_ff = includes_zero or near_zero or wff.get("w") is None

    lines = []
    a = lines.append
    a("# NBA four-factor GAP prototype (2026-11-13)")
    a("")
    a("Simulation only. No wagers were placed. This is a box-score prototype, not a change to the frozen `stack_global` default at `6df88fe`.")
    a("")
    a("## What was built")
    a("- Oliver four factors from team box scores: eFG%, TOV%, ORB%, FT rate, plus FT% and pace.")
    a("- GAP-style **offense and defense**, split by home/away/overall. Venue rates shrink toward the team's overall rate when thin.")
    a("- Exponential daily fade (`forget=0.997`). Ratings used for a game are **as of the prior day**: every game on a date is predicted, then all of that day's boxes update the ratings.")
    a("- Expected PPP is reconstructed from the four factors; `p_home` is a Normal CDF of (PPP margin × pace + 2.5 home edge) / 12.")
    a("- Ratings walk 2023-24 through 2026-27 so the 2025-26 holdout is warm. Scored on the same 1,469 priced Kalshi T-60 games as the calibration / market-blend job.")
    a("- Two leak-free weekly expanding logistic blends, same burn-in as the market-blend job (games before 2025-12-02; 2024-25 stack+Kalshi burn-in was not available):")
    a("  - `four_factors_kalshi_blend`: `a + w_ff logit(FF) + w_k logit(Kalshi)`")
    a("  - `stack_ff_kalshi_blend`: `a + w_s logit(stack_global) + w_ff logit(FF) + w_k logit(Kalshi)` — this is the 'four-factors as an input to stack_global' test, without re-walking the 19-member mixture.")
    a(f"- Code: `{hash_main}` (local). Frozen stack/Elo worktree remains `6df88fe`.")
    a(f"- Walk universe: {len(walk)} completed games with boxes; missing_box={missing_box}; priced with a four-factor probability: {n_ff} / {len(priced)}.")
    a("")
    a("## Relative Ignorance vs Kalshi (bits per game)")
    a("")
    a("Positive `ign_rel_bits` means the model is more ignorant than Kalshi T-60. 90% CIs are the widest of game-level and day-clustered bootstraps.")
    a("")
    a("| model | split | N | ign_rel_bits [90% CI] | growth/event |")
    a("|---|---|---:|---|---:|")
    for key in sorted(gaps, key=lambda k: (k[0], k[1])):
        g = gaps[key]
        a(f"| {key[0]} | {key[1]} | {g['n']} | {g['ign_rel_bits']:+.4f} [{Fmt(g['lo'],4)}, {Fmt(g['hi'],4)}] | {Fmt(g['growth_per_event'],4)} |")
    a("")
    a("## Blend weights (final week, 90% day-clustered bootstrap CI)")
    a("")
    def wline(name, th, key):
        d = th.get(key) or {}
        return f"| {name} | {Fmt(d.get('w'),4)} | [{Fmt(d.get('lo'),4)}, {Fmt(d.get('hi'),4)}] |"
    a("| weight | point | 90% CI |")
    a("|---|---:|---|")
    a(wline("ff+Kalshi  w_ff", th2, "p_ff"))
    a(wline("ff+Kalshi  w_k", th2, "p_mkt"))
    a(wline("3-way  w_stack", th3, "p_stack"))
    a(wline("3-way  w_ff", th3, "p_ff"))
    a(wline("3-way  w_k", th3, "p_mkt"))
    a("")
    ll_ff = ll(have, "p_ff")
    ll_st = ll(have, "p_stack")
    ll_k = ll(have, "p_mkt")
    ll_b2 = ll(scored2, k2)
    ll_b3 = ll(scored3, k3)
    a("## Out-of-sample log loss (nats / bits)")
    a("")
    a("| model | games | nats | bits |")
    a("|---|---:|---:|---:|")
    a(f"| four_factors_gap (full priced set) | {ll_ff[2]} | {Fmt(ll_ff[0],4)} | {Fmt(ll_ff[1],4)} |")
    a(f"| stack_global (same games) | {ll_st[2]} | {Fmt(ll_st[0],4)} | {Fmt(ll_st[1],4)} |")
    a(f"| Kalshi T-60 (same games) | {ll_k[2]} | {Fmt(ll_k[0],4)} | {Fmt(ll_k[1],4)} |")
    a(f"| ff+Kalshi blend (post-burn-in) | {ll_b2[2]} | {Fmt(ll_b2[0],4)} | {Fmt(ll_b2[1],4)} |")
    a(f"| stack+ff+Kalshi blend (post-burn-in) | {ll_b3[2]} | {Fmt(ll_b3[0],4)} | {Fmt(ll_b3[1],4)} |")
    a("")
    a("## Caveats")
    a("- **Injury / availability:** box-score ratings do not adjust for who is out. The market does.")
    a("- This is **not** the PBP `four_factors` member already inside `stack_global`. It is a simpler GAP-style box-score prototype. It was **not** merged into the frozen 19-member walk at `6df88fe`.")
    a("- Home/away splits are shrunk; priors are league-typical, not estimated on a closed tune set.")
    a("- The PPP reconstruction and the Normal margin-to-probability map (sigma=12, home edge=2.5) are untuned defaults.")
    a("- Same Kalshi caveats as the calibration job: no size, home-YES only, Tim's $0.02 fee estimate is not used here because this folder is scoring-only (no new $10k bankroll).")
    a("- Burn-in for the blends is the early 2025-26 holdout (before 2025-12-02), not 2024-25, for the same reason as the market-blend job.")
    a("")
    if pass_ff:
        a("- **Verdict: PASS.** The four-factor weight in the stack+Kalshi blend is about 0 or its 90% CI includes 0, and/or the prototype does not beat Kalshi on relative Ignorance. Do not add it to `stack_global` on this evidence, and do not use it as a betting input.")
    else:
        a("- The three-way `w_ff` CI is above zero on this sample; treat as a research lead only. It has not been merged into the frozen stack and is not a live bet.")
    a("")
    (OUT_DIR / f"{BASE}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dump = {
        "hash_main": hash_main, "hash_wt": hash_wt,
        "n_priced": len(priced), "n_ff": n_ff, "n_walk": len(walk), "missing_box": missing_box,
        "n_scored_ff_mkt": len(scored2), "n_scored_3way": len(scored3),
        "gaps": {f"{m}|{s}": v for (m, s), v in gaps.items()},
        "theta_ff_mkt": th2, "theta_3way": th3,
        "logloss": {"ff": ll_ff, "stack": ll_st, "kalshi": ll_k, "blend_ff": ll_b2, "blend_3way": ll_b3},
        "pass": pass_ff, "elapsed_s": time.time() - t0,
    }
    (OUT_DIR / f"{BASE}_run.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: dump[k] for k in dump if k != "gaps"}, indent=2, default=str), flush=True)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(Main())
    except Exception:
        import traceback
        tb = traceback.format_exc()
        sys.stderr.write(tb); sys.stderr.flush()
        Path(r"E:\AgentWorkspaces\Basketball\basketball\logs\2026-11-13_four-factors.exc").write_text(tb, encoding="utf-8")
        raise