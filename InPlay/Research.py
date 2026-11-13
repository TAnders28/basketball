r"""In-play swing research driver.

    .venv\Scripts\python.exe -m InPlay.Research --stage games     # panel -> Game objects
    .venv\Scripts\python.exe -m InPlay.Research --stage describe
    .venv\Scripts\python.exe -m InPlay.Research --stage calib
    .venv\Scripts\python.exe -m InPlay.Research --stage grid --workers 8   # tuning split only
    .venv\Scripts\python.exe -m InPlay.Research --stage score --workers 8  # selected + pre-registered, holdout/forward

Outputs append to E:\AgentDocs\Basketbot\2026-11-12_inplay-swings\ (csv, _trials.jsonl) and data\inplay\results\.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np

from InPlay.Engine import RunGame, SideMid, TradePnl
from InPlay.Grid import CONTROL_FAMILIES, MAKER_FAMILIES, REPRESENTATIVE, EngineFamily, TrialId, Variants
from InPlay.Load import LoadGames, SplitOf
from InPlay.Stats import CellBootstrap, MaxDrawdown, Pbo, RatioBootstrap, Traversals

OUT_DIR = r"E:\AgentDocs\Basketbot\2026-11-12_inplay-swings"
BASE = "2026-11-12_inplay-swings"
RES = os.path.join("data", "inplay", "results")
BASES = ("tim_est_0.02", "kalshi_taker")
_G = None


def Now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def CsvRows(rows):
    path = os.path.join(OUT_DIR, BASE + ".csv")
    cols = ["section", "venue", "split", "family", "trial_id", "fee_basis", "metric", "value", "lo", "hi", "n", "label"]
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as h:
        if new:
            h.write(",".join(cols) + "\n")
        for r in rows:
            vals = []
            for c in cols:
                x = r.get(c, "")
                if isinstance(x, float):
                    x = "" if np.isnan(x) else f"{x:.6g}"
                x = str(x).replace(",", ";")
                vals.append(x)
            h.write(",".join(vals) + "\n")


def AppendTrials(rows):
    with open(os.path.join(OUT_DIR, BASE + "_trials.jsonl"), "a", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, default=float) + "\n")


def AppendMd(text):
    with open(os.path.join(OUT_DIR, BASE + ".md"), "a", encoding="utf-8") as h:
        h.write(text.rstrip() + "\n\n")


def SaveRes(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name + ".json"), "w", encoding="utf-8") as h:
        json.dump(obj, h, default=float, indent=1)


# ---------------------------------------------------------------- describe
def Describe(G):
    rows, md = [], ["## Descriptive: swings per game (in-play mids)", "",
                    "| venue | pregame fav bucket | games | mean 40/60 traversals | mean 30/70 traversals | share games with >=1 30/70 | mean 0.5 crossings |",
                    "|---|---|---|---|---|---|---|"]
    for venue, games in G.items():
        buckets = defaultdict(list)
        per_q = defaultdict(list)
        for g in games:
            fp = max(g.pre, 1 - g.pre)
            b = "50-55" if fp < 0.55 else "55-60" if fp < 0.60 else "60-65" if fp < 0.65 else "65-75" if fp < 0.75 else "75+"
            t46, _ = Traversals(g.mid, 0.40, 0.60)
            t37, _ = Traversals(g.mid, 0.30, 0.70)
            _, c5 = Traversals(g.mid, 0.50, 0.50 + 1e-9)
            buckets[b].append((t46, t37, c5))
            buckets["all"].append((t46, t37, c5))
            for q in (1, 2, 3, 4):
                seg = [m for m, p in zip(g.mid, g.per) if p == q]
                per_q[q].append(Traversals(seg, 0.40, 0.60)[0])
        for b in ("50-55", "55-60", "60-65", "65-75", "75+", "all"):
            arr = np.array(buckets[b], float) if buckets[b] else np.zeros((0, 3))
            if len(arr) == 0:
                continue
            md.append(f"| {venue} | {b} | {len(arr)} | {arr[:,0].mean():.2f} | {arr[:,1].mean():.2f} | {(arr[:,1]>0).mean():.2f} | {arr[:,2].mean()/2:.1f} |")
            for k, name in ((0, "trav_40_60"), (1, "trav_30_70")):
                rows.append(dict(section="describe", venue=venue, split="all", family="", metric=name + "_per_game",
                                 value=float(arr[:, k].mean()), n=len(arr), label=f"pregame_fav_{b}"))
        for q in (1, 2, 3, 4):
            rows.append(dict(section="describe", venue=venue, split="all", metric="trav_40_60_in_quarter",
                             value=float(np.mean(per_q[q])), n=len(per_q[q]), label=f"Q{q}"))
        md.append("")
        md.append(f"{venue}: mean 40/60 traversals within a single quarter: " +
                  ", ".join(f"Q{q} {np.mean(per_q[q]):.2f}" for q in (1, 2, 3, 4)))
        md.append("")
    CsvRows(rows)
    AppendMd("\n".join(md))
    return rows


# ---------------------------------------------------------------- calibration
def _Obs(games, pick):
    """pick(g, i) -> (side, label) or None. Returns per-game lists of (label, price, outcome)."""
    per_game = []
    for g in games:
        obs = []
        for i in range(g.n):
            m = g.mid[i]
            if m is None:
                continue
            r = pick(g, i)
            if r is None:
                continue
            side, label = r
            p = SideMid(m, side)
            y = g.won if side == 1 else 1 - g.won
            obs.append((label, p, y))
        per_game.append(obs)
    return per_game


def CalTable(per_game, labels, first_only=False):
    """Clustered (game) bootstrap of mean(y - p) per label. first_only keeps one obs per game per label."""
    idx = {l: j for j, l in enumerate(labels)}
    s = np.zeros((len(per_game), len(labels)))
    c = np.zeros_like(s)
    sp = np.zeros_like(s)
    sy = np.zeros_like(s)
    for gi, obs in enumerate(per_game):
        seen = set()
        for label, p, y in obs:
            if label not in idx or (first_only and label in seen):
                continue
            seen.add(label)
            j = idx[label]
            s[gi, j] += y - p
            c[gi, j] += 1
            sp[gi, j] += p
            sy[gi, j] += y
    point, lo, hi = CellBootstrap(s, c, reps=600)
    out = []
    for j, l in enumerate(labels):
        n_obs = c[:, j].sum()
        n_games = (c[:, j] > 0).sum()
        out.append(dict(label=l, n_obs=int(n_obs), n_games=int(n_games),
                        price=float(sp[:, j].sum() / n_obs) if n_obs else float("nan"),
                        win=float(sy[:, j].sum() / n_obs) if n_obs else float("nan"),
                        diff=float(point[j]), lo=float(lo[j]), hi=float(hi[j])))
    return out


def _Bucket(p, w=0.10):
    b = min(int(p / w), int(1 / w) - 1)
    return f"{b*w:.1f}-{(b+1)*w:.1f}"


def Calib(G, splits=("tune", "holdout", "forward")):
    rows, md = [], ["## Calibration in context (P(win | price, state) - price; 95% CI, games resampled)", ""]
    results = {}
    for venue, games_all in G.items():
        games = [g for g in games_all if SplitOf(g.season) in splits]
        # 1) reliability by price bucket x quarter (home side, every bar)
        def p_rel(g, i):
            q = min(g.per[i], 4)
            return (1, f"Q{q}|{_Bucket(g.mid[i])}")
        labels = [f"Q{q}|{_Bucket(b/10+0.05)}" for q in (1, 2, 3, 4) for b in range(10)]
        rel = CalTable(_Obs(games, p_rel), labels)
        # 2) favorite / recent-move contexts on the pregame favorite side, first half
        def p_ctx(g, i):
            fav = 1 if g.pre >= 0.5 else -1
            fp = SideMid(g.pre, fav)
            if g.el[i] > 1440 or fp < 0.55 or fp > 0.65:
                return None
            fm = SideMid(g.mid[i], fav)
            back = g.mid[i - 10] if i >= 10 else None
            tag = "flat"
            if back is not None:
                d = fm - SideMid(back, fav)
                tag = "dropped15_10m" if d <= -0.15 else "rose15_10m" if d >= 0.15 else "flat"
            band = "<=0.40" if fm <= 0.40 else "0.40-0.50" if fm <= 0.50 else ">0.50"
            return (fav, f"fav55-65|H1|{band}|{tag}")
        ctx_labels = [f"fav55-65|H1|{b}|{t}" for b in ("<=0.40", "0.40-0.50", ">0.50") for t in ("dropped15_10m", "flat", "rose15_10m")]
        ctx = CalTable(_Obs(games, p_ctx), ctx_labels)
        # 3) headline: first passage of a 55-65 pregame favorite to <=0.40 in the first half (one obs per game)
        def p_head(g, i):
            fav = 1 if g.pre >= 0.5 else -1
            fp = SideMid(g.pre, fav)
            if 0.55 <= fp <= 0.65 and g.el[i] <= 1440 and SideMid(g.mid[i], fav) <= 0.40:
                return (fav, "HEADLINE fav55-65 first dip <=0.40 in H1")
            return None
        head = CalTable(_Obs(games, p_head), ["HEADLINE fav55-65 first dip <=0.40 in H1"], first_only=True)
        # 4) comeback hypothesis: leader 0.75-0.90 before 6:00 Q4, by pregame closeness (first passage per game)
        def p_cb(g, i):
            if g.el[i] > 2520:
                return None
            m = g.mid[i]
            leader = 1 if m >= 0.5 else -1
            lp = SideMid(m, leader)
            if not (0.75 <= lp <= 0.90):
                return None
            close = abs(g.pre - 0.5)
            fav = 1 if g.pre >= 0.5 else -1
            grp = "even(45-55)" if close <= 0.05 else ("mis(60-80) leader=fav" if leader == fav else "mis(60-80) leader=dog") if 0.10 <= close <= 0.30 else None
            if grp is None:
                return None
            tb = "H1" if g.el[i] <= 1440 else "Q3" if g.el[i] <= 2160 else "Q4<6min-left-excl"
            return (-leader, f"trailer|{grp}|{tb}")
        cb_labels = [f"trailer|{grp}|{tb}" for grp in ("even(45-55)", "mis(60-80) leader=fav", "mis(60-80) leader=dog") for tb in ("H1", "Q3", "Q4<6min-left-excl")]
        cb = CalTable(_Obs(games, p_cb), cb_labels, first_only=True)
        # points-lead version for even games
        def p_cbp(g, i):
            if g.el[i] > 2520 or abs(g.pre - 0.5) > 0.05:
                return None
            sd = g.sd[i]
            if 10 <= abs(sd) <= 18:
                leader = 1 if sd > 0 else -1
                return (-leader, "trailer|even(45-55)|lead10-18pts|before Q4 6:00")
            return None
        cbp = CalTable(_Obs(games, p_cbp), ["trailer|even(45-55)|lead10-18pts|before Q4 6:00"], first_only=True)
        # 5) late favorite-longshot: current favorite in last 6 minutes of regulation, by price
        def p_late(g, i):
            if g.el[i] < 2520 or g.el[i] > 2880:
                return None
            m = g.mid[i]
            fav = 1 if m >= 0.5 else -1
            fm = SideMid(m, fav)
            b = "0.70-0.80" if fm < 0.80 else "0.80-0.90" if fm < 0.90 else "0.90-0.95" if fm < 0.95 else "0.95-0.99" if fm < 0.99 else None
            return None if b is None else (fav, f"lateQ4 fav {b}")
        late_labels = [f"lateQ4 fav {b}" for b in ("0.70-0.80", "0.80-0.90", "0.90-0.95", "0.95-0.99")]
        late = CalTable(_Obs(games, p_late), late_labels)
        results[venue] = dict(n_games=len(games), reliability=rel, context=ctx, headline=head, comeback=cb + cbp, late=late)
        for name, tab in (("reliability", rel), ("context", ctx), ("headline", head), ("comeback", cb + cbp), ("late_fl", late)):
            for r in tab:
                rows.append(dict(section="calib_" + name, venue=venue, split="+".join(splits), metric="win_minus_price",
                                 value=r["diff"], lo=r["lo"], hi=r["hi"], n=r["n_games"], label=r["label"]))
                rows.append(dict(section="calib_" + name, venue=venue, split="+".join(splits), metric="mean_price",
                                 value=r["price"], n=r["n_obs"], label=r["label"]))
        md.append(f"### {venue} ({len(games)} games, splits {'+'.join(splits)})")
        md.append("")
        md.append("| context | games | obs | mean price | win rate | win - price | 95% CI |")
        md.append("|---|---|---|---|---|---|---|")
        for r in head + ctx + cb + cbp + late:
            if r["n_obs"] == 0:
                continue
            md.append(f"| {r['label']} | {r['n_games']} | {r['n_obs']} | {r['price']:.3f} | {r['win']:.3f} | {r['diff']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
        md.append("")
        bad = [r for r in rel if r["n_games"] >= 50 and (r["lo"] > 0 or r["hi"] < 0)]
        md.append(f"Reliability grid (quarter x 0.1 price bucket, all bars): {len([r for r in rel if r['n_games']>=50])} cells with >=50 games; "
                  f"{len(bad)} have a CI excluding 0: " + "; ".join(f"{r['label']} {r['diff']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]" for r in bad))
        md.append("")
    SaveRes("calib_" + "+".join(splits), results)
    CsvRows(rows)
    AppendMd("\n".join(md))
    return results


# ---------------------------------------------------------------- backtests
def _Init():
    global _G
    _G = LoadGames()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _Run(job):
    fam, P, venue, exec_mode, splits = job
    trades, games_by_split, days_by_split = [], defaultdict(int), defaultdict(set)
    for g in _G[venue]:
        sp = SplitOf(g.season)
        if sp not in splits:
            continue
        games_by_split[sp] += 1
        days_by_split[sp].add(g.day)
        for tr in RunGame(EngineFamily(fam), P, g, exec_mode=exec_mode):
            tr["split"] = sp
            trades.append(tr)
    return job, trades, dict(games_by_split), {k: sorted(v) for k, v in days_by_split.items()}


def Metrics(trades, n_games, days, basis):
    days = list(days)
    dix = {d: k for k, d in enumerate(days)}
    pnl_day = np.zeros(len(days))
    n_day = np.zeros(len(days))
    pnls = []
    for tr in sorted(trades, key=lambda t: (t["day"], t["gid"])):
        p = TradePnl(tr, basis)
        pnls.append(p)
        if tr["day"] in dix:
            pnl_day[dix[tr["day"]]] += p
            n_day[dix[tr["day"]]] += 1
    n = len(pnls)
    if n == 0:
        return dict(n_trades=0, n_games=n_games, pnl_per_trade=float("nan"), lo=float("nan"), hi=float("nan"),
                    pnl_per_game=0.0, hit=float("nan"), max_dd=0.0, total=0.0, avg_entry=float("nan"))
    point, lo, hi = RatioBootstrap(pnl_day, n_day, reps=1000)
    pg, pglo, pghi = RatioBootstrap(pnl_day, np.ones(len(days)) * (n_games / max(len(days), 1)), reps=1000)
    return dict(n_trades=n, n_games=n_games, pnl_per_trade=point, lo=lo, hi=hi, pnl_per_game=float(sum(pnls) / max(n_games, 1)),
                pg_lo=pglo, pg_hi=pghi, hit=float(np.mean([p > 0 for p in pnls])), max_dd=MaxDrawdown(pnls),
                total=float(sum(pnls)), avg_entry=float(np.mean([t["entry"] for t in trades])),
                settled_share=float(np.mean([t["settled"] for t in trades])))


def DayPnl(trades, days, basis):
    dix = {d: k for k, d in enumerate(days)}
    out = np.zeros(len(days))
    for tr in trades:
        if tr["day"] in dix:
            out[dix[tr["day"]]] += TradePnl(tr, basis)
    return out


def Jobs():
    jobs = []
    for fam, P in Variants():
        for venue in ("kalshi", "pm"):
            if fam == "xvenue" and venue != "kalshi":
                continue
            jobs.append((fam, P, venue, "taker"))
            if venue == "kalshi" and EngineFamily(fam) in MAKER_FAMILIES and fam != "dip_fav_pace":
                jobs.append((fam, P, venue, "maker"))
    return jobs


def Grid(workers):
    jobs = Jobs()
    t0 = time.time()
    tune = {}
    trial_rows, day_mats = [], defaultdict(list)
    with ProcessPoolExecutor(max_workers=min(workers, 10), initializer=_Init) as ex:
        for (fam, P, venue, xm, _), trades, gsp, dsp in ex.map(_Run, [(*j, ("tune",)) for j in jobs], chunksize=4):
            tid = TrialId(fam, P, venue, xm)
            days = dsp.get("tune", [])
            ms = {b: Metrics(trades, gsp.get("tune", 0), days, b) for b in BASES}
            tune[tid] = dict(family=fam, params=P, venue=venue, exec=xm, tune=ms)
            trial_rows.append(dict(trial_id=tid, created_at=Now(), family=fam, params=P, venue=venue, exec=xm,
                                   control=fam in CONTROL_FAMILIES, split_scored="tune", tune=ms))
            day_mats[(venue, xm)].append((tid, DayPnl(trades, days, "kalshi_taker")))
    print(f"grid: {len(jobs)} trials in {time.time()-t0:.0f}s", flush=True)
    AppendTrials(trial_rows)
    # PBO per venue/exec on per-day P&L (Kalshi taker fee), selectable families only
    pbo = {}
    for key, lst in day_mats.items():
        lst = [(t, d) for t, d in lst if tune[t]["family"] not in CONTROL_FAMILIES]
        mat = np.column_stack([d for _, d in lst]) if lst else np.zeros((0, 0))
        pbo["|".join(key)] = dict(zip(("pbo", "n_combos"), Pbo(mat))) | {"n_variants": len(lst), "n_days": int(mat.shape[0]) if mat.size else 0}
    SaveRes("grid_tune", dict(tune=tune, pbo=pbo))
    return tune, pbo


def Select(tune, min_trades=30):
    """Best variant per family by worst-venue tune P&L per trade (Kalshi taker fee).

    Kalshi tune is the 2025 playoffs only, so a Kalshi requirement uses min_trades//3.
    """
    by_key = defaultdict(dict)
    for tid, t in tune.items():
        key = (t["family"], json.dumps(t["params"], sort_keys=True), t["exec"])
        by_key[key][t["venue"]] = (tid, t["tune"]["kalshi_taker"])
    best = {}
    for (fam, pjson, xm), venues in by_key.items():
        if fam in CONTROL_FAMILIES:
            continue
        scores = []
        for venue, (tid, m) in venues.items():
            need = min_trades // 3 if venue == "kalshi" else min_trades
            if m["n_trades"] < need:
                scores = None
                break
            scores.append(m["pnl_per_trade"])
        if not scores:
            continue
        s = min(scores)
        fam_key = (fam, xm)
        if fam_key not in best or s > best[fam_key][0]:
            best[fam_key] = (s, json.loads(pjson), {v: tid for v, (tid, _) in venues.items()})
    return best


def Score(workers, tune, best):
    jobs, tags = [], {}
    for (fam, xm), (s, P, _) in best.items():
        for venue in ("kalshi", "pm"):
            if fam == "xvenue" and venue != "kalshi":
                continue
            if xm == "maker" and venue != "kalshi":
                continue
            jobs.append((fam, P, venue, xm))
            tags[TrialId(fam, P, venue, xm)] = "selected_on_tune"
    for fam, P in REPRESENTATIVE:
        for venue in ("kalshi", "pm"):
            if fam == "xvenue" and venue != "kalshi":
                continue
            j = (fam, P, venue, "taker")
            tid = TrialId(fam, P, venue, "taker")
            if tid not in tags:
                jobs.append(j)
                tags[tid] = "pre_registered"
            else:
                tags[tid] += "+pre_registered"
    out, rows, trial_rows = {}, [], []
    with ProcessPoolExecutor(max_workers=min(workers, 10), initializer=_Init) as ex:
        for (fam, P, venue, xm, _), trades, gsp, dsp in ex.map(_Run, [(*j, ("tune", "holdout", "forward")) for j in jobs]):
            tid = TrialId(fam, P, venue, xm)
            res = {}
            for sp in ("tune", "holdout", "forward"):
                tr_sp = [t for t in trades if t["split"] == sp]
                res[sp] = {b: Metrics(tr_sp, gsp.get(sp, 0), dsp.get(sp, []), b) for b in BASES}
                for b in BASES:
                    m = res[sp][b]
                    for metric in ("pnl_per_trade", "pnl_per_game", "n_trades", "hit", "max_dd"):
                        lo = m.get("lo") if metric == "pnl_per_trade" else m.get("pg_lo") if metric == "pnl_per_game" else None
                        hi = m.get("hi") if metric == "pnl_per_trade" else m.get("pg_hi") if metric == "pnl_per_game" else None
                        rows.append(dict(section="backtest", venue=venue, split=sp, family=fam, trial_id=tid, fee_basis=b,
                                         metric=metric, value=m.get(metric), lo=lo if lo is not None else "", hi=hi if hi is not None else "",
                                         n=m["n_trades"], label=tags[tid] + "|" + xm))
            out[tid] = dict(family=fam, params=P, venue=venue, exec=xm, tag=tags[tid], res=res)
            trial_rows.append(dict(trial_id=tid, created_at=Now(), family=fam, params=P, venue=venue, exec=xm,
                                   tag=tags[tid], split_scored="tune+holdout+forward", res=res))
    SaveRes("score", out)
    CsvRows(rows)
    AppendTrials(trial_rows)
    return out


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["games", "describe", "calib", "grid", "score"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.stage == "games":
        import pandas as pd
        from InPlay.Load import BuildGames, SaveGames
        G = BuildGames(pd.read_pickle(os.path.join("data", "inplay", "panel.pkl")),
                       pd.read_pickle(os.path.join("data", "inplay", "games.pkl")))
        SaveGames(G)
        for v, gs in G.items():
            c = defaultdict(int)
            for g in gs:
                c[(g.season, SplitOf(g.season))] += 1
            print(v, len(gs), dict(c))
        return
    G = LoadGames()
    if args.stage == "describe":
        Describe(G)
    elif args.stage == "calib":
        Calib(G, ("tune",))
        Calib(G, ("holdout", "forward"))
    elif args.stage == "grid":
        tune, pbo = Grid(args.workers)
        best = Select(tune)
        SaveRes("selected", {f"{k[0]}|{k[1]}": v for k, v in best.items()})
        print(json.dumps(pbo, indent=1))
        for k, v in sorted(best.items(), key=lambda kv: -kv[1][0]):
            print(k, round(v[0], 4), v[1])
    elif args.stage == "score":
        with open(os.path.join(RES, "grid_tune.json"), encoding="utf-8") as h:
            tune = json.load(h)["tune"]
        with open(os.path.join(RES, "selected.json"), encoding="utf-8") as h:
            sel = json.load(h)
        best = {tuple(k.split("|")): tuple(v) for k, v in sel.items()}
        Score(args.workers, tune, best)


if __name__ == "__main__":
    sys.exit(Main())