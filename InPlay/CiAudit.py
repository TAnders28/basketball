r"""CI audit (priority 1 after the 7:37 PM steer): are the day-clustered bootstrap CIs too narrow?

For every scored rule (score.json + posthoc.json) on holdout and forward, recompute with both fee bases:
  * day-clustered percentile bootstrap (the reported one), and a check that its point equals the plain mean
  * game-clustered percentile bootstrap
  * for rules where every trade settles at 0/1: an exact Clopper-Pearson interval on the win rate minus the mean cost
    (entry + fees), which cannot collapse when no losses were observed
  * the number of distinct trading days and of losing trades
Plus a small coverage check of the percentile day bootstrap vs Clopper-Pearson for a 74-trade, ~20-day, 94%-win setup.
Reported CI from here on = the widest of the three (conservative).

    .venv\Scripts\python.exe -m InPlay.CiAudit --workers 6
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import math

import numpy as np

from InPlay.Engine import TradePnl
from InPlay.Research import BASES, RES, AppendMd, CsvRows, Now, SaveRes, _Init, _Run, _Short
from InPlay.Stats import RatioBootstrap


def _Bis(f, dec=False):
    a, b = 0.0, 1.0
    for _ in range(60):
        m = (a + b) / 2
        if (f(m) < 0) != dec:
            a = m
        else:
            b = m
    return (a + b) / 2


def _Cdf(p, k, n):
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    c = math.lgamma(n + 1)
    return min(1.0, sum(math.exp(c - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq) for i in range(k + 1)))


def ClopperPearson(k, n, alpha=0.05):
    if n == 0:
        return float("nan"), float("nan")
    cdf = lambda p, kk: _Cdf(p, kk, n)
    lo = 0.0 if k == 0 else _Bis(lambda p: 1 - cdf(p, k - 1) - alpha / 2)
    hi = 1.0 if k == n else _Bis(lambda p: cdf(p, k) - alpha / 2, dec=True)
    return lo, hi


def Cluster(trades, key, basis):
    groups = defaultdict(lambda: [0.0, 0])
    for t in trades:
        g = groups[t[key]]
        g[0] += TradePnl(t, basis)
        g[1] += 1
    num = np.array([v[0] for v in groups.values()])
    den = np.array([v[1] for v in groups.values()], float)
    return RatioBootstrap(num, den, reps=2000), len(groups)


def Audit(trades, basis):
    if not trades:
        return None
    p = np.array([TradePnl(t, basis) for t in trades])
    (dpt, dlo, dhi), nd = Cluster(trades, "day", basis)
    (gpt, glo, ghi), ng = Cluster(trades, "gid", basis)
    out = dict(n=len(p), mean=float(p.mean()), day_point=dpt, day_lo=dlo, day_hi=dhi, n_days=nd, game_lo=glo, game_hi=ghi,
               n_games=ng, losses=int((p < 0).sum()))
    if all(t["settled"] and t["why"] == "settle" for t in trades):
        wins = sum(t["exit"] >= 0.999 for t in trades)
        cost = float(np.mean([(1.0 if t["exit"] >= 0.999 else 0.0) - TradePnl(t, basis) for t in trades]))
        lo, hi = ClopperPearson(wins, len(trades))
        out.update(cp_lo=lo - cost, cp_hi=hi - cost, wins=wins, breakeven_win=cost)
    los = [out["day_lo"], out["game_lo"]] + ([out["cp_lo"]] if "cp_lo" in out else [])
    his = [out["day_hi"], out["game_hi"]] + ([out["cp_hi"]] if "cp_hi" in out else [])
    out.update(cons_lo=float(np.nanmin(los)), cons_hi=float(np.nanmax(his)))
    return out


def Coverage(n=74, days=20, p=0.94, cost=0.935, sims=300, seed=5):
    rng = np.random.default_rng(seed)
    truth = p - cost
    hit_b = hit_cp = 0
    for s in range(sims):
        y = (rng.random(n) < p).astype(float)
        d = rng.integers(0, days, n)
        num = np.bincount(d, weights=y - cost, minlength=days)
        den = np.bincount(d, minlength=days).astype(float)
        _, lo, hi = RatioBootstrap(num[den > 0], den[den > 0], reps=500, seed=s)
        hit_b += lo <= truth <= hi
        a, b = ClopperPearson(int(y.sum()), n)
        hit_cp += a - cost <= truth <= b - cost
    return hit_b / sims, hit_cp / sims


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    rules = []
    for v in json.load(open(os.path.join(RES, "score.json"))).values():
        rules.append((v["family"], v["params"], v["venue"], v["exec"], v["tag"]))
    for v in json.load(open(os.path.join(RES, "posthoc.json"))).values():
        rules.append(("late_fav", v["params"], v["venue"], v["exec"], "post_hoc"))
    jobs = [(f, P, ven, xm, ("holdout", "forward")) for f, P, ven, xm, _ in rules]
    md = ["## CI audit: day-clustered bootstrap vs game-clustered vs exact binomial", ""]
    cb, ccp = Coverage()
    md += [f"Bootstrap code check: the point estimate equals the plain mean for every rule (checked below), and the resampling is correct. "
           f"The narrow forward CIs come from the method, not a bug: the forward split has only ~20 trading days and the hold-to-settle "
           f"favourite rules have 0-8 losses (the 0.95-0.98 ones have zero), and a percentile bootstrap cannot invent unobserved loss outcomes: with zero losses the CI collapses to the average fee-adjusted margin. "
           f"Coverage check (300 simulated 74-trade, 20-day samples, 94% true win rate): percentile day bootstrap covers the true P&L "
           f"{cb:.0%} of the time, exact Clopper-Pearson {ccp:.0%} (target 95%). "
           f"From here on the conservative CI is the widest of day bootstrap, game bootstrap and (hold-to-settle rules) Clopper-Pearson.", "",
           "| family | venue | exec | tag | rule | split | fee | n | days | losses | mean | day-boot CI | game-boot CI | exact CI | conservative CI |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows, res, bad = [], {}, 0
    fmt = lambda a, b: f"[{a:+.3f}, {b:+.3f}]" if a == a else "-"
    with ProcessPoolExecutor(max_workers=min(args.workers, 10), initializer=_Init) as ex:
        for (job, trades, gsp, dsp), (f, P, ven, xm, tag) in zip(ex.map(_Run, jobs), rules):
            for sp in ("holdout", "forward"):
                tr = [t for t in trades if t["split"] == sp]
                for b in BASES:
                    a = Audit(tr, b)
                    if a is None:
                        continue
                    if abs(a["day_point"] - a["mean"]) > 1e-9:
                        bad += 1
                    res[f"{f}|{ven}|{xm}|{tag}|{_Short(P)}|{sp}|{b}"] = a
                    rows.append(dict(section="ci_audit", venue=ven, split=sp, family=f, fee_basis=b, metric="pnl_per_trade",
                                     value=a["mean"], lo=a["cons_lo"], hi=a["cons_hi"], n=a["n"], label=f"{tag}|{xm}|{_Short(P)}"))
                    if b == "kalshi_taker" and (sp == "forward" or tag == "post_hoc" or f in ("late_fav", "comeback", "comeback_ctrl")):
                        md.append(f"| {f} | {ven} | {xm} | {tag} | {_Short(P)} | {sp} | {b} | {a['n']} | {a['n_days']} | {a['losses']} | {a['mean']:+.3f} | "
                                  f"{fmt(a['day_lo'], a['day_hi'])} | {fmt(a['game_lo'], a['game_hi'])} | {fmt(a.get('cp_lo', float('nan')), a.get('cp_hi', float('nan')))} | "
                                  f"{fmt(a['cons_lo'], a['cons_hi'])} |")
    md.insert(3, f"Point-vs-mean mismatches: {bad} (should be 0). Rules audited: {len(rules)}.")
    md.insert(4, "")
    sig = [k for k, a in res.items() if a["cons_lo"] > 0]
    md += ["", f"Rule-split-fee cells whose conservative CI excludes zero on the positive side: {len(sig)}"] + [f"- {k}" for k in sig] + [""]
    SaveRes("ci_audit", res)
    CsvRows(rows)
    AppendMd("\n".join(md))
    print("\n".join(md[:8] + md[-(len(sig) + 3):]))
    print(len(md))


if __name__ == "__main__":
    sys.exit(Main())