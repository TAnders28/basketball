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

# FINAL SUMMARY (written about 9:40 PM PT, Nov 12, 2026; Rounds 7 and 8 added)

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
- Directional maker off the Polymarket gap: -0.7c (holdout), +2.5c (forward) under the -60 s shift. Round 7A confirms the
  forward number was the shift: with no shift it is -1.18c [-1.68, -0.63] forward and -1.46c [-1.96, -0.95] holdout.
- Calm-stretch-only quoting (low recent volatility, market close to ESPN, no score change; tape-through fills): -0.84c per fill
  [-1.53, -0.15] (holdout), -1.14c [-2.06, -0.35] (forward).
- Last-2:00 possession and timeout cells: the best tune cell ("leader has the ball, trailer has 2+ timeouts") is +1.3c
  [-2.7, +4.9] on Polymarket tune, so it did not qualify and no paper-log entry was added. Holdout: +0.9c (Polymarket) and
  +0.6c (Kalshi), both with CIs around +/-4c.
- 3c+ cross-venue gaps, time to close: under the final run's alignment, Kalshi reaches the Polymarket level in a median of
  20 s (holdout) and 11 s (forward). Without the shift, Polymarket is the slower venue: it reaches the Kalshi mid within 5
  minutes in 59-65% of gaps. Which venue is "slower" depends entirely on the timestamp question.
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

## Round 7 (9:00 PM steer; every variant registered on its tune split before scoring)
**7A. Polymarket timing redone.** Three timings, same rules, same tune selection (Kalshi taker fee): m60 (the earlier -60 s
shift), s0 (no shift; labels on time, as the on-chain audit supports) and perg (per-game offset estimated on each game's first
half against ESPN WP, applied to the second half only).
- Per-game offset: Polymarket moves one minute after ESPN in 93-94% of games in every split (two minutes in 4-5%). It is a
  stable venue/feed lag, not a per-game labelling error. Applying it (perg) means reading Polymarket one minute ahead, which is
  a leak if the labels are on time.
- Every Polymarket-based family stays negative on the holdout under all three timings (e.g. espn_lag on Polymarket -4.7c / -3.6c
  / -3.7c; xvenue on Kalshi -3.1c / -5.7c / -4.6c; Kalshi fee). No holdout CI is above zero under any timing.
- The slower-venue maker (Kalshi maker order when Polymarket disagrees by 3c+; real-print fills; marked to mid 5 min later):

| timing | holdout per fill [95% CI] (fills) | forward per fill [95% CI] (fills) |
|---|---|---|
| m60 (-60 s shift) | -0.73c [-3.20, +2.19] (137) | +2.49c [+0.83, +4.17] (320) |
| s0 (no shift) | -1.46c [-1.96, -0.95] (2,908) | -1.18c [-1.68, -0.63] (2,375) |
| perg (per-game offset, gap 5c) | -0.46c [-1.62, +0.73] (421) | +0.98c [-0.25, +2.30] (465) |

  The only forward positive in the whole study disappears without the shift. It came from seeing Polymarket's next minute.

**7B. Queue-position fills for two-sided quoting (Kalshi tapes, maker fee, marked to mid).** Our quote joins the back of the
queue; it fills only if prints trade through our price or the volume printed at our price exceeds the contracts ahead of us.
Kalshi history has no depth, so the queue ahead is an assumption:

| contracts ahead of us | holdout per fill [95% CI] | forward per fill [95% CI] |
|---|---|---|
| 0 ("touch", optimistic) | +0.19c [+0.16, +0.22] | +0.06c [+0.04, +0.08] |
| 50 | +0.11c [+0.07, +0.13] | -0.04c [-0.08, -0.01] |
| 250 | -0.01c [-0.05, +0.02] | -0.18c [-0.23, -0.13] |
| 1,000 | -0.21c [-0.31, -0.12] | -0.44c [-0.57, -0.31] |
| last 5 min printed at that price | -0.66c [-0.73, -0.59] | -0.91c [-0.95, -0.88] |
| through only (conservative) | -1.20c [-1.33, -1.05] | -1.37c [-1.48, -1.25] |

  The edge is a fraction of a cent even with no queue, and it turns negative once about 50-250 contracts sit ahead of us.
  It also ignores the cost of getting out (it is marked to mid). Not tradable.

**7C. Kalshi moneyline vs Kalshi spread ladder (KXNBASPREAD, free 1-minute candles, 170 forward-window games, 1,896 strike markets).**
- Implied moneyline from a probit fit across strikes (P(home margin > 0.5)) vs the moneyline mid: median gap 1.6c, 90th percentile
  4.1c, gap of 5c or more in 5.1% of in-play minutes.
- Static arbitrage (spread YES bid above moneyline ask, or the mirror): 18 strike-minutes gross, 0 after two taker fees.
- Lead-lag of 1-minute changes: 0.848 at lag 0, 0.003-0.018 at +/-1-2 minutes. Neither market leads the other at minute resolution.
- Trade toward the ladder (selected on early forward games: gap 5c, hold 10 min), scored on later forward games (142 trades):
  -3.2c [-5.6, -1.3] Kalshi fee, -4.8c with $0.02, -0.9c [-3.0, +1.0] with no fee. No holdout: those games are on Kalshi's
  historical endpoint and were not pulled.

**7D. Stars who leave and do not return (ESPN substitutions, 451 Kalshi tape games).** Star = a team's top-2 scorers in its earlier
games (leak-free); event = his last substitution out before 8:00 of Q4 with the game priced 15-85%, not a foul-out.
- Only 20 such exits (1 tune, 13 holdout, 6 forward), against 2,007 ordinary rest exits.
- Holdout: the price keeps moving against the team, -5.9c over 5 minutes [-10.1, -1.6], vs -0.1c after rest exits. Forward: -0.7c
  [-8.8, +9.7]. Too few events to separate from noise, and it did not repeat forward.
- Fading the team 10 minutes: holdout -1.7c [-8.6, +4.7] (13), forward -5.2c [-13.2, +3.8] (6); the control (rest exits) -5.3c / -4.5c.
- Caveat: "does not return" is only known later, so even this is an upper bound. ESPN has no injury flag or team-confirmation
  time, so the under-reaction-until-confirmation question needs news timestamps.

## Round 8 (9:09 PM steer; registered before scoring)
**8A. Our pregame stack_global as the anchor (no-shift data).** Leak-free stack_global probabilities for 2,444 games from
Nov 15, 2024 (`InPlay.StackWalk`: one chronological walk, each member predicts before the result, stack weights refit weekly
on earlier games only; same members and math as PredictDay). Matched to 2,404 Polymarket and 1,378 Kalshi in-play games.
- Our model disagrees with the market's pregame price by 3c or more in about 75% of games (median 7-9c), but the market is
  the better forecaster: holdout log loss market 0.563 vs model 0.594 (Polymarket), 0.568 vs 0.595 (Kalshi); forward 0.540 vs 0.578.
- First-half drift toward our view (|d| >= 3c), signed so positive = toward the model: Polymarket tune +0.06c [-1.3, +1.5],
  holdout -1.1c [-2.3, +0.2], forward -2.5c [-6.2, +1.4]; Kalshi holdout -0.8c [-2.1, +0.6], forward -2.2c [-5.8, +1.6].
  No drift toward our pregame view; if anything the price moves away from it.
- Final outcome vs the market pregame price on the side we prefer: Kalshi holdout -3.1c [-5.7, -0.4]. Where we disagree,
  the market is closer.
- Trade (selected on Polymarket tune: gap 5c, hold to settle): holdout -5.1c (Polymarket) / -5.4c (Kalshi) with Kalshi's fee,
  CIs [-8.4, -1.6] / [-8.7, -2.0]; forward -12.2c / -13.3c (72 / 80 trades). With $0.02: -5.7c / -6.1c holdout. Every
  first-quarter and halftime exit variant also lost on tune (-3.7 to -4.3c).

**8B. Kalshi live totals vs a simple pace projection (KXNBATOTAL, 1,422 strike markets, 168 forward-window games).**
- The market median total beats the pace projection at every checkpoint: mean absolute error 12.7 vs 13.9 points (end of Q1),
  10.9 vs 12.3 (Q2), 7.7 vs 8.1 (Q3).
- Does the pace-minus-market gap predict the final-minus-market miss? Slope -0.08 [-0.47, +0.34] (Q1), -0.10 [-0.47, +0.27] (Q2),
  +0.27 [-0.15, +0.71] (Q3). All CIs include 0: the market already prices pace.
- Trade toward pace (selected on early forward games: gap 3 points at halftime), later forward games: -15.0c [-28.0, -0.8] per
  trade with Kalshi's fee (54 trades), -15.3c with $0.02. No 2025-26 holdout (historical endpoint not pulled).

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
   Round 7A reran every Polymarket-based rule with no shift and with per-game offsets: Polymarket trails ESPN by one minute in
   94% of games, every Polymarket rule still loses, and the Round 6A forward positive vanishes without the shift (it was a
   one-minute look-ahead). Treat the no-shift results as the honest ones.
2. Polymarket has no historical book. Its spread is an assumed 1c, and its fees and access were not verified; it is scored
   with the two Kalshi-style fee bases.
3. The Kalshi tune split is only 86 playoff games, so Kalshi-only selection is noisy. Polymarket tune (1,267 games) carries
   most of the selection.
4. Kalshi fee: the formula is applied unrounded per contract. With 1-lot orders, Kalshi's round-up makes costs slightly worse.
   Tim's $0.02 is an estimate.
5. Maker fills: Kalshi history has no book depth, so queue position is an assumption. Round 7B shows any quoting edge is under
   0.2c with no queue and negative with 50-250 contracts ahead; the "through" model is the conservative bound.
6. ESPN WP is a model, not a price, and nothing on ESPN is fillable.
7. Forward is about 3 weeks (170 games).
8. Rule breach disclosed: earlier in this run one shell command read a terminal log outside E:\\ (C:/Users/.../terminals).
   That has not been repeated.

## Next steps
1. Paper-score the frozen spec weekly with `InPlay.PaperScore` (no tuning).
2. Optional: a live side-by-side capture (Polymarket CLOB websocket vs Kalshi, a few games, paper only) would confirm the
   one-minute Polymarket feed lag directly. Use the no-shift data meanwhile.
5. Star-exit idea: needs injury news timestamps (team PR / beat-writer posts) to test under-reaction until confirmation.
3. Done in Round 8A: the PredictDay stack as anchor loses; the market's pregame price is the better forecaster. Improve the
   pregame model before reusing it as an in-play anchor.
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