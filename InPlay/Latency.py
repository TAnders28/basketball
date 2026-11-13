r"""Diagnostic (not a selection step): how much do latency and execution matter for the fast signals?

Runs the pre-registered fast-signal rules on Kalshi holdout+forward with latency 0/1/2 minutes, frictionless (mid fills,
no fees) and Kalshi taker. Latency 0 means filling at the same bar that produced the signal: an unreachable upper bound.

    .venv\Scripts\python.exe -m InPlay.Latency
"""

import sys

import numpy as np

from InPlay.Engine import Game, RunGame, TradePnl
from InPlay.Grid import REPRESENTATIVE
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, CsvRows, RatioBootstrap, _Short

FAST = ("momentum", "fade_run", "espn_lag", "xvenue", "dip_any")


def Main():
    G = LoadGames()
    games = [g for g in G["kalshi"] if SplitOf(g.season) in ("holdout", "forward")]
    mids = []
    for g in games:
        h = Game(**{k: getattr(g, k) for k in Game.__slots__ if k != "n"})
        h.bid, h.ask = list(g.mid), list(g.mid)
        mids.append(h)
    md = ["## Diagnostic: latency sensitivity of fast signals (Kalshi, holdout+forward, pre-registered rules)", "",
          "| rule | latency (min) | trades | frictionless P/L per trade [95% CI] | Kalshi-taker P/L per trade |", "|---|---|---|---|---|"]
    rows = []
    for fam, P in REPRESENTATIVE:
        if fam not in FAST:
            continue
        for lat in (0, 1, 2):
            fr, net = [], []
            for g, h in zip(games, mids):
                fr += [(g.day, TradePnl(t, "none")) for t in RunGame(fam, P, h, lat=lat, slip=0.0)]
                net += [TradePnl(t, "kalshi_taker") for t in RunGame(fam, P, g, lat=lat)]
            days = sorted({d for d, _ in fr})
            ix = {d: k for k, d in enumerate(days)}
            num, den = np.zeros(len(days)), np.zeros(len(days))
            for d, p in fr:
                num[ix[d]] += p
                den[ix[d]] += 1
            pt, lo, hi = RatioBootstrap(num, den, reps=800)
            md.append(f"| {fam} {_Short(P)} | {lat} | {len(fr)} | {pt:+.4f} [{lo:+.4f}, {hi:+.4f}] | {np.mean(net):+.4f} |")
            rows.append(dict(section="latency_diag", venue="kalshi", split="holdout+forward", family=fam, fee_basis="none_mid_fills",
                             metric="pnl_per_trade", value=pt, lo=lo, hi=hi, n=len(fr), label=f"lat={lat}"))
            rows.append(dict(section="latency_diag", venue="kalshi", split="holdout+forward", family=fam, fee_basis="kalshi_taker",
                             metric="pnl_per_trade", value=float(np.mean(net)), n=len(net), label=f"lat={lat}"))
    CsvRows(rows)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())