r"""Kalshi trade tape checks (priority 3 of the 7:49 PM steer). Needs InPlay.PullTrades output.

A. Maker-quoting fill model vs real prints. Same rule as Round3 mm_quote (quote the YES bid and ask seen at minute bar i,
   1 contract each), but a fill now requires a real print in (t_i, t_i + 60 s] by a taker on the other side:
   our bid fills on a taker SELL of YES (taker_side = no) at or through our bid; our ask on a taker BUY of YES at or
   through our ask. Two queue assumptions: "through" (print strictly better for the taker than our level, i.e. the
   level was exhausted: conservative) and "touch" (any print at our level: optimistic, ignores queue). Marked to mid
   H minutes later, Kalshi maker fee. Compared with the candle-extreme model on the same games.
B. Sub-minute response to scoring plays. For each in-play scoring play with |ESPN WP change| >= 0.02, direction d,
   pre price = last print before (wall - 30 s); response r(s) = d * (last print at wall + s - pre). Share of the 180 s
   move done by each offset, split by free throws (dead ball) vs field goals (live play). ESPN wallclock error and
   market lag cannot be separated here: the offsets are relative to ESPN's own timestamp.

    .venv\Scripts\python.exe -m InPlay.TradeTape
"""

import bisect
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from InPlay.Align import EspnStates
from InPlay.Execution import Fee
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, RatioBootstrap
from InPlay.Round3 import MmFills
from InPlay.Stats import CellBootstrap

TR = os.path.join("data", "inplay", "kalshi", "trades")
OFFS = (-30, -15, 0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180)


def LoadTape(ticker):
    p = os.path.join(TR, ticker + ".json")
    if not os.path.exists(p):
        return None
    tr = json.load(open(p))["trades"]
    if not tr:
        return None
    a = np.array(tr, float)
    return a[np.argsort(a[:, 0], kind="stable")]


def TapeFills(g, tape, X, H, tmax, mode):
    ts, px, side = tape[:, 0], tape[:, 1], tape[:, 3]
    out = []
    for i in range(1, g.n - 1):
        el = g.el[i]
        if el < 0 or el > tmax:
            continue
        b, a, m0, m1 = g.bid[i], g.ask[i], g.mid[i], g.mid[i - 1]
        if b is None or a is None or m0 is None or a - b < 0.01:
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
        lo, hi = bisect.bisect_right(ts, g.t[i]), bisect.bisect_right(ts, g.t[i] + 60)
        if hi <= lo:
            continue
        w_px, w_sd = px[lo:hi], side[lo:hi]
        sells, buys = w_px[w_sd < 0], w_px[w_sd > 0]
        eps = 1e-9
        hit_bid = (sells < b - eps).any() if mode == "through" else (sells <= b + eps).any()
        lift_ask = (buys > a + eps).any() if mode == "through" else (buys >= a - eps).any()
        if hit_bid:
            out.append((g.day, mark - b - Fee(b, "kalshi_maker"), m0 - b, mark - m0))
        if lift_ask:
            out.append((g.day, a - mark - Fee(a, "kalshi_maker"), a - m0, m0 - mark))
    return out


def Score(fills, n_games):
    if not fills:
        return dict(n=0, per_fill=float("nan"), lo=float("nan"), hi=float("nan"), half_spread=float("nan"), adverse=float("nan"), per_game=0.0)
    days = sorted({f[0] for f in fills})
    ix = {d: k for k, d in enumerate(days)}
    num, den = np.zeros(len(days)), np.zeros(len(days))
    for d, p, _, _ in fills:
        num[ix[d]] += p
        den[ix[d]] += 1
    pt, lo, hi = RatioBootstrap(num, den, reps=800)
    return dict(n=len(fills), per_fill=pt, lo=lo, hi=hi, half_spread=float(np.mean([f[2] for f in fills])),
                adverse=float(np.mean([f[3] for f in fills])), per_game=float(sum(f[1] for f in fills) / max(n_games, 1)))


def EventStudy(game_json, tape):
    ts, px = tape[:, 0], tape[:, 1]
    st = EspnStates(game_json)
    out = []
    for k in range(1, len(st)):
        a, b = st[k - 1], st[k]
        if b["wp"] is None or a["wp"] is None or (b["hs"] == a["hs"] and b["as"] == a["as"]):
            continue
        if b["el"] > 2820 or b["per"] > 4:
            continue
        dw = b["wp"] - a["wp"]
        if abs(dw) < 0.02:
            continue
        w = b["t"]
        j0 = bisect.bisect_right(ts, w - 30) - 1
        if j0 < 0:
            continue
        p0 = px[j0]
        r = []
        for s in OFFS:
            j = bisect.bisect_right(ts, w + s) - 1
            r.append((px[j] - p0) * (1 if dw > 0 else -1))
        kind = "free throw (dead ball)" if "free throw" in (b.get("type") or "").lower() else "field goal (live)"
        out.append((kind, r, abs(dw)))
    return out


def Main():
    G = LoadGames()
    meta = pd.read_pickle(os.path.join("data", "inplay", "games.pkl")).dropna(subset=["kalshi_ticker"])
    tick = dict(zip(meta.game_id.astype(str), meta.kalshi_ticker))
    seas = dict(zip(meta.game_id.astype(str), meta.season))
    games = [(g, tick.get(str(g.gid))) for g in G["kalshi"]]
    games = [(g, t, LoadTape(t)) for g, t in games if t]
    games = [(g, t, tp) for g, t, tp in games if tp is not None and len(tp) > 50]
    md = ["## Kalshi trade tape checks (public trade prints, second timestamps)", "",
          f"Games with a tape: {len(games)} (" + ", ".join(f"{sp}: {sum(SplitOf(g.season) == sp for g, _, _ in games)}" for sp in ("tune", "holdout", "forward")) + ")", ""]
    rows, trials = [], []
    # A
    md += ["### A. Maker quoting: candle-extreme fill model vs real prints (same games)", "",
           "| variant | split | fill model | fills | P/L per fill, maker fee [95% CI] | half-spread earned | adverse move | P/L per game |", "|---|---|---|---|---|---|---|---|"]
    for X, lab in ((0.02, "pull=0.02 (selected on tune)"), (None, "no pull")):
        P = dict(pull=X, mark_h=5, tmax=2160)
        for sp in ("tune", "holdout", "forward"):
            gs = [(g, tp) for g, _, tp in games if SplitOf(g.season) == sp]
            if not gs:
                continue
            for model in ("candle", "tape-through", "tape-touch"):
                fills = []
                for g, tp in gs:
                    fills += MmFills(g, X, 5, 2160) if model == "candle" else TapeFills(g, tp, X, 5, 2160, model.split("-")[1])
                r = Score(fills, len(gs))
                md.append(f"| {lab} | {sp} | {model} | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
                rows.append(dict(section="tape_mm", venue="kalshi", split=sp, family="mm_quote", fee_basis="kalshi_maker", metric="pnl_per_fill",
                                 value=r["per_fill"], lo=r["lo"], hi=r["hi"], n=r["n"], label=f"{lab}|{model}"))
                trials.append(dict(trial_id=TrialId("mm_quote_tape", dict(P, model=model), "kalshi", "maker_both"), created_at=Now(),
                                   family="mm_quote_tape", params=dict(P, model=model), venue="kalshi", exec="maker_both",
                                   tag="tape_check_of_round3_rule", split_scored=sp, res=r))
    md.append("")
    # B
    ev = defaultdict(list)
    for g, t, tp in games:
        p = os.path.join("data", "inplay", "espn", "games", str(seas[str(g.gid)]), f"{g.gid}.json")
        if not os.path.exists(p):
            continue
        for kind, r, dw in EventStudy(json.load(open(p, encoding="utf-8")), tp):
            ev[kind].append((g.gid, r, dw))
    md += ["### B. Kalshi price move (cents, in the direction of the ESPN WP change) vs seconds after ESPN's play timestamp; game-clustered 95% CI", "",
           "| play type | plays | mean abs ESPN WP change | " + " | ".join(f"{s:+d}s" for s in OFFS) + " |",
           "|---|---|---|" + "---|" * len(OFFS)]
    for kind, lst in sorted(ev.items()):
        gids = sorted({x[0] for x in lst})
        gi = {x: k for k, x in enumerate(gids)}
        S = np.zeros((len(gids), len(OFFS)))
        C = np.zeros_like(S)
        for gid, r, _ in lst:
            S[gi[gid]] += r
            C[gi[gid]] += 1
        pt, lo, hi = CellBootstrap(S, C, reps=400)
        full = pt[-1]
        cells = [f"{pt[k]*100:.2f} [{lo[k]*100:.2f}, {hi[k]*100:.2f}]" for k in range(len(OFFS))]
        md.append(f"| {kind} | {len(lst)} | {np.mean([x[2] for x in lst]):.3f} | " + " | ".join(cells) + " |")
        rem = {s: full - pt[k] for k, s in enumerate(OFFS)}
        md.append(f"| {kind}: move still to come before +180 s (c) | | | " + " | ".join(f"{rem[s]*100:.2f}" for s in OFFS) + " |")
        for k, s in enumerate(OFFS):
            rows.append(dict(section="tape_event_study", venue="kalshi", split="all_tape_games", family="describe", metric="signed_move",
                             value=float(pt[k]), lo=float(lo[k]), hi=float(hi[k]), n=len(lst), label=f"{kind}|{s:+d}s"))
    md += ["", "Reading: a taker who sees ESPN's play at offset s and pays the half-spread plus the taker fee (about 0.5c + 1.7c at p = 0.5) "
           "needs the move still to come after s to exceed about 2.2c on average; the second table row shows it.", ""]
    AppendTrials(trials + [dict(trial_id="tape-event-study", created_at=Now(), family="tape_event_study", tag="descriptive",
                                params=dict(offsets=OFFS, min_dwp=0.02), split_scored="all_tape_games")])
    CsvRows(rows)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())