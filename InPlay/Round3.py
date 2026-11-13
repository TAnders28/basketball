r"""Round 3 (after the 7:37 PM steer; registered after first holdout scoring).

A. Two-sided maker quoting on Kalshi with an adverse-selection "pull" filter. Each bar we quote 1 contract at the
   YES bid and 1 at the YES ask seen at bar i. A quote fills only if the next bar's trades print strictly through it.
   Each fill is marked to the mid H bars later (or settled if the game ends first) and pays the Kalshi maker fee.
   Pull: skip quoting when |change in ESPN WP| or |change in mid| over the last bar exceeds X. Optimistic: no queue
   position, unlimited inventory, candle trade extremes instead of a trade tape.
B. Does market-vs-ESPN disagreement predict future volatility (not direction)? Descriptive.
C. End-of-game microstructure: last 2:00 of Q4, leader by 3-8, leader price 0.80-0.97; calibration by margin.
D. Freeze a forward paper registry (no tuning after this): rules to score only on games after 2026-11-12.

    .venv\Scripts\python.exe -m InPlay.Round3
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

from InPlay.Execution import Fee
from InPlay.Grid import REPRESENTATIVE, TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import RES, AppendMd, AppendTrials, CalTable, CsvRows, Now, RatioBootstrap, SaveRes, _Obs, _Short
from InPlay.Stats import CellBootstrap

PAPER_START = "2026-11-13"


def MmFills(g, X, H, tmax):
    out = []
    for i in range(1, g.n - 1):
        el = g.el[i]
        if el < 0 or el > tmax:
            continue
        b, a, m0, m1 = g.bid[i], g.ask[i], g.mid[i], g.mid[i - 1]
        if b is None or a is None or m0 is None or a - b < 0.01 or g.trlo[i + 1] is None:
            continue
        if X is not None:
            dm = abs(m0 - m1) if m1 is not None else 0.0
            dw = abs(g.wp[i] - g.wp[i - 1]) if g.wp[i] is not None and g.wp[i - 1] is not None else 0.0
            if max(dm, dw) > X:
                continue
        j = i + 1 + H
        if j >= g.n:
            mark = float(g.won)
        else:
            mark = next((g.mid[k] for k in range(j, max(i, j - 5), -1) if g.mid[k] is not None), None)
            if mark is None:
                continue
        if g.trlo[i + 1] < b - 1e-9:
            out.append((g.day, mark - b - Fee(b, "kalshi_maker"), (m0 - b), mark - m0))
        if g.trhi[i + 1] > a + 1e-9:
            out.append((g.day, a - mark - Fee(a, "kalshi_maker"), (a - m0), m0 - mark))
    return out


def MmScore(games, X, H, tmax):
    fills = []
    for g in games:
        fills += MmFills(g, X, H, tmax)
    if not fills:
        return dict(n=0)
    days = sorted({f[0] for f in fills})
    ix = {d: k for k, d in enumerate(days)}
    num, den = np.zeros(len(days)), np.zeros(len(days))
    for d, p, _, _ in fills:
        num[ix[d]] += p
        den[ix[d]] += 1
    pt, lo, hi = RatioBootstrap(num, den, reps=800)
    return dict(n=len(fills), per_fill=pt, lo=lo, hi=hi, half_spread=float(np.mean([f[2] for f in fills])),
                adverse=float(np.mean([f[3] for f in fills])), per_game=float(sum(f[1] for f in fills) / len(games)))


def Main():
    G = LoadGames()
    md, rows, trials = ["## Round 3 (after the 7:37 PM steer; registered after first holdout scoring)", ""], [], []
    K = G["kalshi"]
    by = {sp: [g for g in K if SplitOf(g.season) == sp] for sp in ("tune", "holdout", "forward")}
    # ---- A
    grid = [(X, H, tm) for X in (None, 0.02, 0.04) for H in (1, 5, 15) for tm in (2880, 2160)]
    tune = {}
    for X, H, tm in grid:
        P = dict(pull=X, mark_h=H, tmax=tm)
        r = MmScore(by["tune"], X, H, tm)
        tid = TrialId("mm_quote", P, "kalshi", "maker_both")
        tune[tid] = (P, r)
        trials.append(dict(trial_id=tid, created_at=Now(), family="mm_quote", params=P, venue="kalshi", exec="maker_both",
                           tag="round3_after_first_holdout_scoring", split_scored="tune", tune=r))
    elig = [(tid, P, r) for tid, (P, r) in tune.items() if r.get("n", 0) >= 100]
    tid, Pb, _ = max(elig, key=lambda x: x[2]["per_fill"])
    md += [f"### A. Two-sided maker quoting with adverse-selection pull (Kalshi, {len(grid)} variants on tune; best scored once)", "",
           "Per fill, marked to mid H minutes later, net of the Kalshi maker fee. Optimistic fill model (no queue, no inventory limit).", "",
           "| variant | split | fills | P/L per fill [95% CI] | half-spread earned | adverse move | P/L per game |", "|---|---|---|---|---|---|---|"]
    for P, r in sorted(tune.values(), key=lambda x: -x[1].get("per_fill", -9))[:4]:
        md.append(f"| {_Short(P)} | tune | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
    res = {}
    for sp in ("holdout", "forward"):
        for lab, P in (("selected", Pb), ("no-pull baseline", dict(pull=None, mark_h=Pb["mark_h"], tmax=Pb["tmax"]))):
            r = MmScore(by[sp], P["pull"], P["mark_h"], P["tmax"])
            res[f"{lab}|{sp}"] = r
            md.append(f"| {lab}: {_Short(P)} | {sp} | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
            rows.append(dict(section="round3_mm", venue="kalshi", split=sp, family="mm_quote", trial_id=TrialId("mm_quote", P, "kalshi", "maker_both"),
                             fee_basis="kalshi_maker", metric="pnl_per_fill", value=r["per_fill"], lo=r["lo"], hi=r["hi"], n=r["n"], label=f"{lab}: {_Short(P)}"))
    trials.append(dict(trial_id=tid, created_at=Now(), family="mm_quote", params=Pb, venue="kalshi", exec="maker_both",
                       tag="round3_selected_on_tune", split_scored="holdout+forward", res=res))
    md.append("")
    # ---- B
    md += ["### B. Does |market - ESPN WP| predict the next 10 minutes of price movement? (descriptive)", "",
           "| venue | split | disagreement | bars | mean abs 10-min mid change [95% CI, game clusters] |", "|---|---|---|---|---|"]
    edges = (0, 0.03, 0.06, 0.10, 0.20, 1.01)
    for venue in ("kalshi", "pm"):
        for sp in ("tune", "holdout"):
            gs = [g for g in G[venue] if SplitOf(g.season) == sp]
            s = np.zeros((len(gs), len(edges) - 1))
            c = np.zeros_like(s)
            for gi, g in enumerate(gs):
                for i in range(g.n - 10):
                    if g.mid[i] is None or g.mid[i + 10] is None or g.wp[i] is None or not (0 <= g.el[i] <= 2280):
                        continue
                    d = abs(g.mid[i] - g.wp[i])
                    k = next(k for k in range(len(edges) - 1) if d < edges[k + 1])
                    s[gi, k] += abs(g.mid[i + 10] - g.mid[i])
                    c[gi, k] += 1
            pt, lo, hi = CellBootstrap(s, c, reps=400)
            for k in range(len(edges) - 1):
                md.append(f"| {venue} | {sp} | {edges[k]:.2f}-{min(edges[k+1],1):.2f} | {int(c[:, k].sum())} | {pt[k]:.4f} [{lo[k]:.4f}, {hi[k]:.4f}] |")
                rows.append(dict(section="round3_disagree_vol", venue=venue, split=sp, family="describe", metric="mean_abs_10min_change",
                                 value=float(pt[k]), lo=float(lo[k]), hi=float(hi[k]), n=int(c[:, k].sum()), label=f"{edges[k]:.2f}-{edges[k+1]:.2f}"))
    md.append("")
    # ---- C
    labels = ["lead 3", "lead 4-5", "lead 6-8"]

    def Pick(g, i):
        if g.per[i] != 4 or not (2760 <= g.el[i] <= 2880) or g.sd[i] is None:
            return None
        side = 1 if g.sd[i] > 0 else -1
        a = abs(g.sd[i])
        p = g.mid[i] if side == 1 else 1 - g.mid[i]
        if not (0.80 <= p <= 0.97) or a < 3 or a > 8:
            return None
        return side, ("lead 3" if a == 3 else "lead 4-5" if a <= 5 else "lead 6-8")

    md += ["### C. Last 2:00 of Q4, leader by 3-8, leader priced 0.80-0.97: leader win rate minus price (first bar per game per cell)", "",
           "| venue | split | cell | games | price | win | win-price [95% CI] |", "|---|---|---|---|---|---|---|"]
    for venue in ("pm", "kalshi"):
        for sps in (("tune",), ("holdout", "forward")):
            gs = [g for g in G[venue] if SplitOf(g.season) in sps]
            for r in CalTable(_Obs(gs, Pick), labels, first_only=True):
                if r["n_games"]:
                    md.append(f"| {venue} | {'+'.join(sps)} | {r['label']} | {r['n_games']} | {r['price']:.3f} | {r['win']:.3f} | {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
                    rows.append(dict(section="round3_endgame_calib", venue=venue, split="+".join(sps), family="calib", metric="win_minus_price",
                                     value=r["diff"], lo=r["lo"], hi=r["hi"], n=r["n_games"], label=r["label"]))
    md.append("")
    # ---- D
    sel = json.load(open(os.path.join(RES, "selected.json")))
    frozen = [dict(family=k.split("|")[0], exec=k.split("|")[1], params=v[1], trial_ids=v[2], why="selected on tune") for k, v in sel.items() if k.endswith("|taker")]
    frozen += [dict(family=f, exec="taker", params=P, why="pre-registered representative") for f, P in REPRESENTATIVE if f == "late_fav"]
    frozen += [dict(family="late_fav", exec="maker", params=dict(mins=12, lo=0.95, hi=0.98), why="post-hoc favorite-longshot candidate (holdout contaminated)")]
    reg = dict(frozen_at=Now(), score_games_on_or_after_et=PAPER_START, venues=["kalshi", "pm"], fees=["tim_est_0.02", "kalshi_taker"],
               note="Paper only. No tuning allowed after this file; score each rule once per review on games after the start date.", rules=frozen)
    with open(os.path.join("data", "inplay", "paper_registry.json"), "w") as f:
        json.dump(reg, f, indent=1)
    trials.append(dict(trial_id="paper-registry-2026-11-12", created_at=Now(), family="paper_registry", tag="frozen_forward_paper",
                       params=reg, split_scored="none"))
    md += [f"### D. Frozen forward paper registry (`data\\inplay\\paper_registry.json`): {len(frozen)} rules, scored only on games from {PAPER_START} ET, no further tuning", ""]
    md += [f"- {r['family']} ({r['exec']}): {_Short(r['params'])} [{r['why']}]" for r in frozen]
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round3", dict(mm_tune={k: v for k, v in tune.items()}, mm_scored=res))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())