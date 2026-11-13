r"""Compact holdout/forward table for the final summary (reads data\inplay\results\score.json; no new trials)."""

import json
import os
import sys

from InPlay.Research import RES, _Short


def Cell(m):
    if not m or not m.get("n_trades"):
        return "-"
    return f"{m['pnl_per_trade']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}] (n={m['n_trades']})"


def Table(keep):
    d = json.load(open(os.path.join(RES, "score.json")))
    out = ["| family | venue | exec | tag | rule | holdout P/L per trade, Kalshi fee [95% CI] | holdout P/L per trade, Tim est. | holdout P/L per game | holdout hit / max DD | forward P/L per trade, Kalshi fee [95% CI] |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for v in sorted(d.values(), key=lambda v: (v["family"], v["venue"], v["exec"], v["tag"])):
        if not keep(v):
            continue
        h, f = v["res"].get("holdout", {}), v["res"].get("forward", {})
        hk, ht = h.get("kalshi_taker"), h.get("tim_est_0.02")
        if not hk or not hk.get("n_trades"):
            continue
        out.append(f"| {v['family']} | {v['venue']} | {v['exec']} | {v['tag'].replace('_', ' ')} | {_Short(v['params'])} | {Cell(hk)} | {ht['pnl_per_trade']:+.3f} | "
                   f"{hk['pnl_per_game']:+.4f} | {hk['hit']:.2f} / {hk['max_dd']:.1f} | {Cell(f.get('kalshi_taker'))} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(Table(lambda v: v["exec"] == "taker" and (v["tag"] == "selected_on_tune" or v["family"] in ("dip_fav", "comeback", "late_fav"))))