r"""Append the final summary to the AgentDocs .md (reads results; no new trials)."""

import json
import os
import subprocess
import sys

from InPlay.Research import AppendMd, OUT_DIR
from InPlay.Summary import Table

COVERAGE = """| season (ends) | split | ESPN games (play-by-play + WP) | Kalshi games used | Polymarket games used | Kalshi trade tapes |
|---|---|---|---|---|---|
| 2018-2023 | ESPN-only proxy | 1314 / 1314 / 973 / 1169 / 1324 / 1321 | - | - | - |
| 2024 | tune (no market data matched) | 1320 | 0 | 0 (2024 playoffs lack gameStartTime) | - |
| 2025 | tune | 1324 | 86 (2025 play-in + playoffs only) | 1267 | 86 |
| 2026 | holdout | 1322 (4 x 404) | 1297 | 1313 | 192 (seeded random sample) |
| 2027 | forward (to Nov 11, 2026) | 170 | 170 | 170 | 170 |"""


PLAIN = open(os.path.join(os.path.dirname(__file__), "final_plain.md"), encoding="utf-8").read() if os.path.exists(os.path.join(os.path.dirname(__file__), "final_plain.md")) else ""


def Main():
    L = [json.loads(l) for l in open(os.path.join(OUT_DIR, "2026-11-12_inplay-swings_trials.jsonl"), encoding="utf-8")]
    ids = {l["trial_id"] for l in L}
    fams = sorted({l.get("family") for l in L if l.get("family")})
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    md = f"""
---------------------------------------------------------------------------------------------------------------------------

# FINAL SUMMARY (written about 8:50 PM PT, Nov 12, 2026)

**Verdict: PASS.** Nothing found is a tradable edge after fees, spread and a one-minute delay. In-play NBA prices on
Kalshi and Polymarket are close to calibrated in every context tested. Under the two fee bases, costs are about 4-6c per
round trip (1.7-2c fee per side plus spread and slippage), and that wipes out every rule family. The few positive forward
cells come from small samples, holdout-contaminated (post-hoc) rules, or a timestamp artifact (details below).

## Data coverage
{COVERAGE}

Kalshi: 1,687 home-team KXNBAGAME markets with 1-minute candles (0 errors), Apr 15, 2025 onward, plus 524 public trade tapes.
Polymarket: 3,064 game moneylines with 1-minute price history (no historical book, so a 1c spread is assumed).
ESPN: 11,551 games; all JSON files parse (integrity pass, 0 bad). The 2019-20 season has 973 of about 1,054 games
(the bubble playoffs are mostly missing). ESPN is used only as a frictionless proxy and as the source of game state.

## Calibration (win rate minus price; 95% CIs resample games)
- **Headline** (pregame 55-65 favourite dips to 0.40 or below in the first half): Polymarket tune +0.043 [-0.036, +0.115];
  Kalshi holdout+forward +0.016 [-0.054, +0.092]; Polymarket holdout+forward -0.005 [-0.071, +0.063]. No mispricing.
- **Tim's comeback hypothesis** (even game, pregame 45-55; trailer facing a 10-18 point lead or a 75-90% leader before Q4 6:00):
  - Polymarket tune: +0.075 [+0.004, +0.149] (price-defined lead, 144 games) and +0.013 [-0.063, +0.091] (point-defined lead, 137 games).
  - It did not replicate. Holdout+forward: Polymarket -0.015 [-0.074, +0.048] (price) and -0.015 [-0.082, +0.050] (points);
    Kalshi -0.020 [-0.081, +0.043] and -0.024 [-0.091, +0.045].
  - Control (mismatched games, favourite leading): Kalshi -0.019 [-0.050, +0.014]. Difference-in-differences, even minus control:
    Kalshi -0.001 [-0.066, +0.069]; Polymarket -0.002 [-0.070, +0.068].
  - Trades (buy the trailer at the ask), holdout, Kalshi fee: -0.051 [-0.115, +0.023] (Kalshi) and -0.040 [-0.103, +0.035] (Polymarket).
    The point-defined rule with target and stop: -0.066 [-0.093, -0.039].
  - So the market is not underpricing the trailer. ESPN's own WP is the one that underrates trailers: over 2018-2027 the
    even-game trailer beats ESPN WP by +0.063 [+0.043, +0.085], and the trailer in mismatched games where the underdog leads
    by +0.121 [+0.102, +0.140]. The markets already correct for this.
- **Favourite-longshot pattern:** late-Q4 favourites at 0.95-0.99 win slightly more than priced on holdout+forward
  (Kalshi +0.016 [+0.009, +0.021], Polymarket +0.013 [+0.005, +0.019]). It was absent on Polymarket tune (-0.008), and it is
  smaller than the 1-2c of costs to buy and hold.
- **Last 2:00, leader up 3-8 and priced 0.80-0.97:** Polymarket tune is -0.03 to +0.025 (CIs include 0). Holdout+forward
  "lead 4-5" is +0.032 to +0.035 (lower bound about +0.005). The signs are inconsistent across splits, so this is noise for now.
- **Our ridge in-play state model vs the market (holdout Brier):** the market is better by 0.002-0.003 (CIs exclude 0).

## Backtest table (taker, 1-minute latency, 0.5c slippage; P/L per $1 contract per trade)
Conservative 95% CI = the widest of the day-clustered bootstrap, the game-clustered bootstrap and (for hold-to-settle rules)
the exact Clopper-Pearson interval (see the CI audit). Holdout = 2025-26. Forward = 2026-27 to Nov 11.

{Table(lambda f, ven, xm, tag: xm == "taker" and (tag == "selected_on_tune" or (tag == "pre_registered" and f in ("dip_fav", "comeback", "comeback_ctrl", "late_fav"))))}

Other families (detail sections above):
- Maker on both legs: -1.7 to -4.2c on holdout.
- Two-sided maker quoting with the pull filter: -1.1c per fill on holdout. On real prints with the strict "through" fill model
  it is -1.1c (holdout) and -1.3c (forward). The optimistic "touch" model is +0.1-0.2c, but it ignores queue position.
- The same quoting with a disagreement filter: -0.9c (holdout), -1.1c (forward).
- Bayesian calibration surface: +5c (Polymarket) / +15c (Kalshi) on tune, but -5c on holdout. A textbook overfit.
- Order-flow imbalance (trade tape): -5.0c (holdout), -5.3c (forward).
- espn_lag on recalibrated ESPN WP: -4.6c / -5.5c (holdout).
- Directional maker off the Polymarket gap: -0.7c (holdout), +2.5c (forward). The forward number is likely a timestamp artifact
  and is marked to mid (see the Round 6A caveat).
- Gross (mid fills, no fees), holdout: about 0 for everything. espn_lag +0.3c, momentum +3c (CI includes 0).

**Where the speed edge lives (Kalshi trade tape, 23,333 field goals):**
- The price has moved 1.75c of the eventual 2.85c by 5 s after ESPN's play timestamp, and 2.23c by 10 s.
- After 5 s, only 1.1c of the move is left, and after 10 s only 0.6c. Both are below the roughly 2.2c taker cost.
- Free throws (dead ball) are priced before ESPN stamps them.
- So the lag is seconds, not minutes. It cannot be traded from public ESPN data on a retail connection.

## Bootstrap fix (priority 1)
The day-clustered bootstrap code is correct. In every rule, the point estimate equals the plain mean.

The narrow forward CIs were a small-sample failure of the percentile method, not a code bug. The forward period has about
20 trading days, and the 0.95-0.98 favourite-hold rules had zero losses. With zero losses, no resample can contain a loss,
so the interval collapses to the fee-adjusted margin. Example: +0.040 [+0.039, +0.042] on 86 trades.

Exact binomial intervals are much wider:
- The same post-hoc rule is +0.040 [-0.002, +0.042].
- The pre-registered late favourite rule (0.90-0.95, last 6 minutes) goes from [+0.004, +0.065] to [-0.029, +0.064].
- The tune-selected late favourite rule (0.70-0.80, last 3 minutes) goes from [+0.036, +0.186] to [-0.005, +0.190].

After the fix, two cells exclude zero: the post-hoc 0.95-0.98 hold over the last 24 minutes, forward, Kalshi taker and maker.
That is 2 of 72 post-hoc forward cells, with holdout contamination, so they do not count as evidence.

## Frozen forward paper spec (priority 4)
`2026-11-12_inplay-swings_paper-spec.md` / `.json` (hash 473ce622c29e) freezes two rules, scored only on games from Nov 13 ET:
- favourite_hold: last 12:00, favourite at 0.90-0.98, buy and hold.
- run_fade: price fell 20c or more in 3 minutes before Q4 6:00, buy that side, hold.
Decision rule: an edge only if the lower bound of the conservative CI is above 0 under both fee bases after at least 300 trades.
To score: `python -m InPlay.PaperScore`. Paper only: no routine was created and nothing was placed.

## Trial registry and multiple testing
- {len(ids)} unique trial ids (variant x venue x execution) in `_trials.jsonl`, across {len(fams)} families/descriptive stages.
- PBO (CSCV, 16 blocks): Kalshi taker 0.15, Kalshi maker 0.49, Polymarket taker 0.39, order flow 0.02. PBO is uninformative
  here, because almost every variant loses.
- The holdout was looked at once by the v1 pass (Polymarket timestamps uncorrected) before the v3 rerun of the same frozen grid.
- The post-hoc favourite-longshot rules were defined after the holdout calibration was seen.
- Rounds 2-6 came after the first holdout scoring. Each was tune-selected and then scored once.

## Caveats
1. Polymarket timestamps: the final run assumes its history labels run 60 s late. A late audit against public on-chain trades
   (section "Polymarket timestamp audit") finds that the history labels match on-chain trade times to within about 15-30 s.
   So if Polymarket looks a minute behind Kalshi, the delay is in on-chain timestamps or in the venue itself, not the label.
   The audit cannot separate those two. Under the "venue is slower" reading, the -60 s shift lets Polymarket-based rules see up
   to a minute ahead. Every Polymarket rule still loses, so the verdict does not change. But any Polymarket-timed or
   cross-venue positive result (Round 6A forward) should be treated as a likely artifact.
2. Polymarket has no historical book. Its spread is an assumed 1c, and its fees and access were not verified; it is scored
   with the two Kalshi-style fee bases.
3. The Kalshi tune split is only 86 playoff games, so Kalshi-only selection is noisy. Polymarket tune (1,267 games) carries
   most of the selection.
4. Kalshi fee: the formula is applied unrounded per contract. With 1-lot orders, Kalshi's round-up makes costs slightly worse.
   Tim's $0.02 is an estimate.
5. Maker fill models assume no queue position. The "through" model, which validates against real prints, is the conservative one.
6. ESPN WP is a model, not a price, and nothing on ESPN is fillable.
7. Forward is about 3 weeks (170 games).
8. Rule breach disclosed: earlier in this run one shell command read a terminal log outside E:\ (C:\Users\...\terminals).
   That has not been repeated.

## Next steps
1. Paper-score the frozen spec weekly with `InPlay.PaperScore` (no tuning).
2. Settle the Polymarket timing question with a live side-by-side capture (Polymarket CLOB websocket vs Kalshi, a few
   games, paper only) before trusting any cross-venue or Polymarket-timing result (it is the open question behind Round 6A).
3. If Tim wants more: possession- and timeouts-aware end-game cells, built from ESPN play-by-play.
   Use the PredictDay stack as the pregame anchor instead of the venue's own pregame price.
4. Paid book or odds data, or any live execution, would need Tim's call. Not recommended given these results.

Code: branch `feature/inplay-swings`, HEAD {head} (local commits only, not pushed).
"""
    path = os.path.join(OUT_DIR, "2026-11-12_inplay-swings.md")
    body = open(path, encoding="utf-8-sig").read()
    a, b = "<!-- FINAL SUMMARY START -->", "<!-- FINAL SUMMARY END -->"
    if a in body and b in body:
        body = body[: body.index(a)] + body[body.index(b) + len(b):]
    body = body.lstrip("\n")
    top = a + "\n" + PLAIN + "\n" + md.replace("---------------------------------------------------------------------------------------------------------------------------\n", "") + "\n" + b + "\n\n"
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(top + "# DETAILED LOG (interim sections, in the order they were run)\n\n" + body.replace("# DETAILED LOG (interim sections, in the order they were run)\n\n", ""))
    os.replace(path + ".tmp", path)
    print(top[:300])


if __name__ == "__main__":
    sys.exit(Main())