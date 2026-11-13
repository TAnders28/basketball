r"""Frozen forward paper-trading spec (written 2026-11-12, about 8:10 PM PT). Paper only: no orders, no wagers, no routines.

Two rules, one version each, scored only on games whose ET date is on or after 2026-11-13. No parameter may change after
this file is committed; any change is a new spec with a new date. Scoring re-pulls data with the existing pullers and runs:

    .venv\Scripts\python.exe -m InPlay.PaperScore          (prints the scorecard; refuses games before the start date)
    .venv\Scripts\python.exe -m InPlay.PaperScore --write  (also writes the spec md/json into the AgentDocs folder)
"""

import argparse
import hashlib
import json
import os
import sys

from InPlay.CiAudit import Audit
from InPlay.Engine import RunGame
from InPlay.Load import LoadGames

START_ET = "2026-11-13"
OUT = r"E:\AgentDocs\Basketbot\2026-11-12_inplay-swings"
SPEC = {
    "spec_date": "2026-11-12",
    "score_games_on_or_after_et": START_ET,
    "venue": "Kalshi KXNBAGAME (home-YES series; the away side is bought as NO = 1 - YES); Polymarket scored as a shadow",
    "fills": "taker: 1 contract at the side ask of the NEXT 1-minute bar after the signal bar (1-minute latency) + 0.5c slippage",
    "fees": ["tim_est_0.02 (Tim's working estimate, $0.02 per contract per side)", "kalshi_taker (roundup(0.07*C*P*(1-P)), no settlement fee)"],
    "rules": [
        {"name": "favorite_hold", "family": "late_fav", "params": {"mins": 12, "lo": 0.90, "hi": 0.98},
         "plain": "In the last 12:00 of regulation, the first minute bar where the current favorite's price is 0.90-0.98: buy the favorite, hold to settlement. One trade per game.",
         "origin": "POST-HOC: suggested by the holdout calibration (favorite-longshot); holdout is contaminated, so only these forward games count."},
        {"name": "run_fade", "family": "fade_run", "params": {"drop": 0.20, "hold": None, "look": 3, "max_tr": 3, "tmax": 2520},
         "plain": "Before 6:00 of Q4, when a side's price has fallen 20c or more over the last 3 minutes and is inside 0.10-0.90: buy that side, hold to settlement (up to 3 per game).",
         "origin": "Selected on the tune split (worst-season rule); holdout -0.046 per trade (Kalshi fee), forward +0.024 on 54 trades before this spec."},
    ],
    "decision_rule": "Report P&L per trade with the conservative CI (widest of day bootstrap, game bootstrap, exact binomial). "
                     "Only call an edge if the lower bound is above 0 under BOTH fee bases after at least 300 trades; otherwise PASS.",
}
SPEC_HASH = hashlib.sha1(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()[:12]


def Scorecard():
    G = LoadGames()
    lines = [f"Spec {SPEC_HASH}, games on or after {START_ET} ET", "", "| rule | venue | games | trades | fee | P/L per trade [conservative 95% CI] |", "|---|---|---|---|---|---|"]
    for r in SPEC["rules"]:
        for venue in ("kalshi", "pm"):
            gs = [g for g in G[venue] if g.day >= START_ET]
            tr = []
            for g in gs:
                assert g.day >= START_ET
                tr += RunGame(r["family"], r["params"], g)
            for b in ("tim_est_0.02", "kalshi_taker"):
                a = Audit(tr, b)
                cell = f"{a['mean']:+.3f} [{a['cons_lo']:+.3f}, {a['cons_hi']:+.3f}]" if a else "no trades yet"
                lines.append(f"| {r['name']} | {venue} | {len(gs)} | {len(tr)} | {b} | {cell} |")
    return "\n".join(lines)


def WriteSpec():
    base = os.path.join(OUT, "2026-11-12_inplay-swings_paper-spec")
    with open(base + ".json", "w") as fh:
        json.dump(dict(SPEC, spec_hash=SPEC_HASH), fh, indent=1)
    md = [f"# Frozen forward paper spec (2026-11-12, hash {SPEC_HASH})", "",
          "Paper only. Nothing is placed, no routine is created. Scored only on games from " + START_ET + " ET onward; no tuning after this date.", "",
          f"- Venue: {SPEC['venue']}", f"- Fills: {SPEC['fills']}", "- Fees: " + "; ".join(SPEC["fees"]), ""]
    for r in SPEC["rules"]:
        md += [f"## {r['name']} ({r['family']}: {json.dumps(r['params'])})", "", r["plain"], "", "Origin: " + r["origin"], ""]
    md += ["## Decision rule", "", SPEC["decision_rule"], "", "## How to score", "",
           "Pull the new games with the existing resumable pullers (InPlay.PullEspn / PullKalshi / PullPolymarket), rebuild the panel "
           "(InPlay.BuildPanel, InPlay.Research --stage games), then run `.venv\\Scripts\\python.exe -m InPlay.PaperScore`.", "",
           "## Scorecard at spec time", "", Scorecard(), ""]
    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return base


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.write:
        print(WriteSpec())
    print(Scorecard())


if __name__ == "__main__":
    sys.exit(Main())