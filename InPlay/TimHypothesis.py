r"""Tim's comeback hypothesis, consolidated (calibration with game-clustered CIs, control, and the trade results).

Hypothesis: in evenly matched games (pregame 45-55%), when one team leads by 10-18 points or is priced 75-90% before
6:00 of Q4, the market underprices the trailer. Control: the same situation in mismatched games (pregame 60-80% for one
side), split by whether the leader is the pregame favourite or the underdog.

First passage per game per cell (the first bar at which the game enters the cell); trailer win rate minus trailer price;
95% CI resampling games. Difference-in-differences vs the control uses a paired game bootstrap over the pooled games.
The comeback trade rules were registered and tune-selected earlier (families comeback / comeback_ctrl); this module adds
no new trade trials. The holdout calibration for this hypothesis was first seen in the 7:2x PM scoring run.

    .venv\Scripts\python.exe -m InPlay.TimHypothesis
"""

import json
import os
import sys

import numpy as np

from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import RES, AppendMd, AppendTrials, CsvRows, Now

GROUPS = ("even", "mis_leader_fav", "mis_leader_dog")
LEADS = ("pts10-18", "price75-90")


def Group(pre, leader):
    c = abs(pre - 0.5)
    if c <= 0.05:
        return "even"
    if 0.10 <= c <= 0.30:
        fav = 1 if pre >= 0.5 else -1
        return "mis_leader_fav" if leader == fav else "mis_leader_dog"
    return None


def FirstPassage(g):
    out = {}
    if g.pre is None:
        return out
    for i in range(g.n):
        m, el, sd = g.mid[i], g.el[i], g.sd[i]
        if m is None or el is None or el <= 0 or el > 2520:
            continue
        for lead in LEADS:
            if lead == "pts10-18":
                if sd is None or not (10 <= abs(sd) <= 18):
                    continue
                leader = 1 if sd > 0 else -1
            else:
                leader = 1 if m >= 0.5 else -1
                lp = m if leader == 1 else 1 - m
                if not (0.75 <= lp <= 0.90):
                    continue
            grp = Group(g.pre, leader)
            if grp is None or (grp, lead) in out:
                continue
            tp = 1 - m if leader == 1 else m
            ty = (1 - g.won) if leader == 1 else g.won
            out[(grp, lead)] = (tp, ty)
    return out


def Boot(rows_by_game, reps=2000, seed=21):
    """rows_by_game: list of dict cell -> (p, y). Returns per-cell (n, price, win, diff, lo, hi) and DiD per lead."""
    cells = [(g, l) for g in GROUPS for l in LEADS]
    G = len(rows_by_game)
    S = np.zeros((G, len(cells)))
    C = np.zeros_like(S)
    P = np.zeros_like(S)
    Y = np.zeros_like(S)
    for gi, d in enumerate(rows_by_game):
        for j, c in enumerate(cells):
            if c in d:
                p, y = d[c]
                S[gi, j], C[gi, j], P[gi, j], Y[gi, j] = y - p, 1, p, y
    rng = np.random.default_rng(seed)
    W = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(reps)]).astype(float)
    bs = (W @ S) / np.maximum(W @ C, 1e-12)
    out = {}
    for j, c in enumerate(cells):
        n = C[:, j].sum()
        if n == 0:
            continue
        out[c] = dict(n=int(n), price=float(P[:, j].sum() / n), win=float(Y[:, j].sum() / n), diff=float(S[:, j].sum() / n),
                      lo=float(np.quantile(bs[:, j], 0.025)), hi=float(np.quantile(bs[:, j], 0.975)))
    did = {}
    for l in LEADS:
        je, jf = cells.index(("even", l)), cells.index(("mis_leader_fav", l))
        if C[:, je].sum() and C[:, jf].sum():
            pt = S[:, je].sum() / C[:, je].sum() - S[:, jf].sum() / C[:, jf].sum()
            d = bs[:, je] - bs[:, jf]
            did[l] = dict(diff=float(pt), lo=float(np.quantile(d, 0.025)), hi=float(np.quantile(d, 0.975)))
    return out, did


def Main():
    G = LoadGames()
    md = ["## Tim's hypothesis: evenly matched games, trailer facing a 10-18 point / 75-90% lead before Q4 6:00", "",
          "Trailer win rate minus trailer price, first passage per game; 95% CI resampling games. Control = mismatched games "
          "(pregame 60-80% for one side). DiD = even minus mismatch-with-favourite-leading.", "",
          "| venue | split | lead definition | group | games | trailer price | trailer win | win - price [95% CI] |", "|---|---|---|---|---|---|---|---|"]
    rows, res = [], {}
    did_lines = []
    for venue in ("pm", "kalshi"):
        for sps in (("tune",), ("holdout",), ("forward",), ("holdout", "forward")):
            gs = [g for g in G[venue] if SplitOf(g.season) in sps]
            if not gs:
                continue
            cells, did = Boot([FirstPassage(g) for g in gs])
            name = "+".join(sps)
            for (grp, lead), v in cells.items():
                md.append(f"| {venue} | {name} | {lead} | {grp} | {v['n']} | {v['price']:.3f} | {v['win']:.3f} | {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
                rows.append(dict(section="tim_hypothesis_calib", venue=venue, split=name, family="comeback_calib", metric="trailer_win_minus_price",
                                 value=v["diff"], lo=v["lo"], hi=v["hi"], n=v["n"], label=f"{grp}|{lead}"))
                res[f"{venue}|{name}|{grp}|{lead}"] = v
            for lead, v in did.items():
                did_lines.append(f"| {venue} | {name} | {lead} | {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
                rows.append(dict(section="tim_hypothesis_did", venue=venue, split=name, family="comeback_calib", metric="did_even_minus_mismatch_fav",
                                 value=v["diff"], lo=v["lo"], hi=v["hi"], label=lead))
                res[f"{venue}|{name}|did|{lead}"] = v
    md += ["", "| venue | split | lead definition | DiD (even - mismatch, favourite leads) [95% CI] |", "|---|---|---|---|"] + did_lines
    audit = json.load(open(os.path.join(RES, "ci_audit.json")))
    md += ["", "Trade results for the registered comeback rules (buy the trailer at the ask, 1-minute latency, 0.5c slippage); "
           "conservative CI = widest of day bootstrap, game bootstrap and exact binomial (see CI audit).", "",
           "| family | venue | tag | rule | split | fee basis | trades | P/L per trade [conservative 95% CI] | losses |", "|---|---|---|---|---|---|---|---|---|"]
    for k, a in sorted(audit.items()):
        f, ven, xm, tag, rule, sp, b = k.split("|")
        if f in ("comeback", "comeback_ctrl") and xm == "taker":
            md.append(f"| {f} | {ven} | {tag} | {rule} | {sp} | {b} | {a['n']} | {a['mean']:+.3f} [{a['cons_lo']:+.3f}, {a['cons_hi']:+.3f}] | {a['losses']} |")
    md.append("")
    AppendTrials([dict(trial_id="tim-hypothesis-calib-consolidated", created_at=Now(), family="comeback_calib",
                       tag="descriptive_consolidation_after_holdout_seen", params=dict(groups=GROUPS, leads=LEADS, tmax=2520),
                       split_scored="tune,holdout,forward", res=res)])
    CsvRows(rows)
    AppendMd("\n".join(md))
    with open(os.path.join(RES, "tim_hypothesis.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())