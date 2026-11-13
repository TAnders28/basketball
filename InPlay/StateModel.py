r"""Round-2 family: our own Bayesian (ridge-logistic) in-play state model vs the market.

Trained only on ESPN play-by-play outcomes from seasons <= 2024 (no market data, no tuning-split games),
features: logit(pregame prob), score diff scaled by sqrt(time left), time-decayed pregame logit, raw diff, possession-free.
A Gaussian prior (L2 ridge) on the coefficients is the Bayesian part; the posterior mode is used.
In backtests the market's own pregame price is the pregame input. Trades use the espn_lag engine with the
model probability in place of ESPN WP: buy the side the model says is underpriced by >= gap, exit on convergence/time.
This family was registered after the first holdout scoring (round 2), so it is labelled as such.

    .venv\Scripts\python.exe -m InPlay.StateModel --workers 8
"""

import argparse
import glob
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from InPlay.Align import EspnStates
from InPlay.Engine import Game, RunGame
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import BASES, AppendMd, AppendTrials, CsvRows, Metrics, Now, RatioBootstrap, SaveRes, _Short

TRAIN_SEASONS = range(2018, 2025)


def Logit(p):
    p = min(0.995, max(0.005, p))
    return math.log(p / (1 - p))


def Features(pre, el, sd):
    rem = max(0.0, 2880.0 - el) / 60.0  # minutes left in regulation (OT -> 0)
    lp = Logit(pre)
    z = sd / math.sqrt(rem + 1.0)
    return [1.0, lp, lp * rem / 48.0, z, sd / 10.0, math.tanh(z / 2.0)]


def _Rows(path):
    g = json.load(open(path, encoding="utf-8"))
    st = [s for s in EspnStates(g) if s["wp"] is not None]
    if len(st) < 100 or g.get("home_won") is None:
        return None
    pre = st[0]["wp"]
    y = int(bool(g["home_won"]))
    out = []
    for s in st[::8]:
        if s["el"] > 2880:
            continue
        out.append(Features(pre, s["el"], s["hs"] - s["as"]) + [y])
    return out


def Sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1 / (1 + math.exp(-z))


def Fit(X, y, lam=1.0, iters=30):
    w = np.zeros(X.shape[1])
    prior = np.full(X.shape[1], lam)
    prior[0] = 1e-6
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + prior * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prior)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def ModelGames(G, w):
    out = {}
    for venue, games in G.items():
        lst = []
        for g in games:
            h = Game(**{k: getattr(g, k) for k in Game.__slots__ if k != "n"})
            h.wp = [Sigmoid(float(np.dot(w, Features(g.pre, e, s)))) for e, s in zip(g.el, g.sd)]
            lst.append(h)
        out[venue] = lst
    return out


_MG = None


def _InitM(w):
    global _MG
    import ctypes
    if os.name == "nt":
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    _MG = ModelGames(LoadGames(), np.array(w))


def _RunM(job):
    P, venue, splits = job
    trades, gsp, dsp = [], {}, {}
    for g in _MG[venue]:
        sp = SplitOf(g.season)
        if sp not in splits:
            continue
        gsp[sp] = gsp.get(sp, 0) + 1
        dsp.setdefault(sp, set()).add(g.day)
        for tr in RunGame("espn_lag", P, g):
            tr["split"] = sp
            trades.append(tr)
    return job, trades, gsp, {k: sorted(v) for k, v in dsp.items()}


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    paths = [p for s in TRAIN_SEASONS for p in glob.glob(os.path.join("data", "inplay", "espn", "games", str(s), "*.json"))]
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, 10)) as ex:
        for r in ex.map(_Rows, paths, chunksize=25):
            if r:
                rows += r
    A = np.array(rows)
    X, y = A[:, :-1], A[:, -1]
    w = Fit(X, y)
    n_games = len(paths)
    # calibration of the model on the tuning split (market pregame input), all bars, Brier vs market mid
    G = LoadGames()
    M = ModelGames(G, w)
    brier = {}
    for venue, games in M.items():
        for sp in ("tune", "holdout", "forward"):
            d_g, n_g, bm, bk = [], [], 0.0, 0.0
            for g in games:
                if SplitOf(g.season) != sp:
                    continue
                dm = dn = 0.0
                for p, m in zip(g.wp, g.mid):
                    if m is None:
                        continue
                    bm += (p - g.won) ** 2
                    bk += (m - g.won) ** 2
                    dm += (p - g.won) ** 2 - (m - g.won) ** 2
                    dn += 1
                d_g.append(dm)
                n_g.append(dn)
            if sum(n_g):
                pt, lo, hi = RatioBootstrap(d_g, n_g, reps=1000)
                brier[f"{venue}|{sp}"] = dict(n_bars=int(sum(n_g)), n_games=len(n_g), brier_model=bm / sum(n_g),
                                              brier_market=bk / sum(n_g), diff=pt, lo=lo, hi=hi)
    variants = []
    for gap in (0.04, 0.06, 0.08, 0.12):
        for hold in (3, 10, None):
            for tmax in (2160, 2820):
                variants.append(dict(gap=gap, dir=1, hold=hold, close_gap=True, tmax=tmax, max_tr=3))
    jobs = [(P, v, ("tune",)) for P in variants for v in ("kalshi", "pm")]
    tune = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, 10), initializer=_InitM, initargs=(list(w),)) as ex:
        for (P, venue, _), trades, gsp, dsp in ex.map(_RunM, jobs):
            tune[(json.dumps(P, sort_keys=True), venue)] = Metrics(trades, gsp.get("tune", 0), dsp.get("tune", []), "kalshi_taker")
    # select on tune: worst of PM tune and (if >= 10 trades) Kalshi tune
    best, best_s = None, -9
    for P in variants:
        key = json.dumps(P, sort_keys=True)
        s = []
        pm_m, k_m = tune[(key, "pm")], tune[(key, "kalshi")]
        if pm_m["n_trades"] >= 30:
            s.append(pm_m["pnl_per_trade"])
        if k_m["n_trades"] >= 10:
            s.append(k_m["pnl_per_trade"])
        if s and min(s) > best_s:
            best, best_s = P, min(s)
    trials = [dict(trial_id=TrialId("statemodel", P, v, "taker"), created_at=Now(), family="statemodel", params=P, venue=v,
                   exec="taker", tag="round2_after_first_holdout_scoring", split_scored="tune",
                   tune={"kalshi_taker": tune[(json.dumps(P, sort_keys=True), v)]}) for P in variants for v in ("kalshi", "pm")]
    # score the selected variant once
    res = {}
    with ProcessPoolExecutor(max_workers=2, initializer=_InitM, initargs=(list(w),)) as ex:
        for (P, venue, _), trades, gsp, dsp in ex.map(_RunM, [(best, v, ("tune", "holdout", "forward")) for v in ("kalshi", "pm")]):
            res[venue] = {}
            for sp in ("tune", "holdout", "forward"):
                tr = [t for t in trades if t["split"] == sp]
                res[venue][sp] = {b: Metrics(tr, gsp.get(sp, 0), dsp.get(sp, []), b) for b in BASES}
            trials.append(dict(trial_id=TrialId("statemodel", best, venue, "taker"), created_at=Now(), family="statemodel",
                               params=best, venue=venue, exec="taker", tag="round2_selected_on_tune", split_scored="tune+holdout+forward",
                               res=res[venue]))
    AppendTrials(trials)
    csv = []
    md = ["## Round 2: our own Bayesian in-play state model vs the market", "",
          f"Ridge-logistic (Gaussian prior, lambda=1) fitted on {len(rows)} ESPN play states from {n_games} games, seasons 2017-18 to 2023-24",
          "(outcomes only, no market data). Coefficients [1, logit(pre), logit(pre)*t_left/48, z=diff/sqrt(min_left+1), diff/10, tanh(z/2)]: " +
          ", ".join(f"{x:+.3f}" for x in w), "",
          "Brier, all in-play bars, model (market pregame price as input) minus market mid; 95% CI resampling games. Negative = model better: " +
          "; ".join(f"{v}: {b['brier_model']:.4f} vs {b['brier_market']:.4f}, diff {b['diff']:+.4f} [{b['lo']:+.4f}, {b['hi']:+.4f}] ({b['n_games']} games)" for v, b in brier.items()), "",
          f"{len(variants)*2} variants tried on tune; selected {_Short(best)} (tune worst-venue P&L/trade {best_s:+.4f}, kalshi_taker).", "",
          "| venue | split | fee basis | trades | P/L per trade [95% CI] | hit |", "|---|---|---|---|---|---|"]
    for venue, r in res.items():
        for sp in ("tune", "holdout", "forward"):
            for b in BASES:
                m = r[sp][b]
                if m["n_trades"]:
                    md.append(f"| {venue} | {sp} | {b} | {m['n_trades']} | {m['pnl_per_trade']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] | {m['hit']:.2f} |")
                    csv.append(dict(section="statemodel_backtest", venue=venue, split=sp, family="statemodel",
                                    trial_id=TrialId("statemodel", best, venue, "taker"), fee_basis=b, metric="pnl_per_trade",
                                    value=m["pnl_per_trade"], lo=m["lo"], hi=m["hi"], n=m["n_trades"], label=_Short(best)))
    for v, b in brier.items():
        venue, sp = v.split("|")
        csv.append(dict(section="statemodel_brier", venue=venue, split=sp, metric="brier_model_minus_market", value=b["diff"],
                        lo=b["lo"], hi=b["hi"], n=b["n_games"]))
    CsvRows(csv)
    AppendMd("\n".join(md))
    SaveRes("statemodel", dict(w=list(w), brier=brier, best=best, res=res))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())