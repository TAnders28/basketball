# In-play NBA swings - plain-language findings (final, Nov 12, 2026, about 9:35 PM PT)

**Verdict: PASS. Do not trade any of these rules.** We registered about 2,000 strategy variants. Each was tuned on 2024-25,
then scored once on 2025-26 and on 2026-27 to date. The market prices behaved like nearly calibrated probabilities. In-game
swings are common, but buying dips or fading runs loses money once you pay the spread, fees, slippage and a one-minute delay.
Typical cost is about 4-6c per contract per round trip. Every rule family is negative or indistinguishable from zero on the
holdout, under both fee scenarios (Tim's $0.02-per-side estimate, and Kalshi's published taker fee).

- **Tim's evenly matched comeback idea (pregame 45-55%, leader then at 75-90% or up 10-18 before late Q4):** the pattern is in
  ESPN's win-probability model, not in the market. Against ESPN WP the trailer wins 6.3 points more often than predicted
  (2018-2027, 95% CI +4.3 to +8.5). Against market prices the trailer is not underpriced on the holdout plus forward split:
  - Kalshi: -2.0 points (CI -8.1 to +4.3).
  - Polymarket: -1.5 points (CI -7.4 to +4.8).
  - Compared with the mismatched-game control: about 0 (Kalshi -0.1, CI -6.6 to +6.9).
  - Buying the trailer at the ask lost 4-5c per contract on the holdout (Kalshi -5.1c, CI -11.5 to +2.3; Polymarket -4.0c).
  - The +7.5-point tune-split result on Polymarket did not replicate.
- **Market-making (Kalshi):** checked against real Kalshi trade prints, and fill only if prints trade through our price. Quoting
  both sides loses about 1.1c per fill on the holdout, even at the lower maker fee:
  - Pulling quotes after ESPN or price jumps only narrows the loss.
  - Quoting only when the market agrees with ESPN only narrows it.
  - Quoting only in calm stretches still loses: -0.8c per fill on the holdout, -1.1c forward.
  - Being picked off (about 1.2-1.6c) costs more than the half-spread earned (about 0.5-0.6c).
  - Filling every order that merely touches our price shows +0.1-0.2c, but that ignores queue position.
- **Sub-minute lag:** on 23,333 made field goals, Kalshi had moved 1.75c of the full 2.85c move 5 seconds after ESPN's play
  timestamp, and 2.2c after 10 seconds. After 10 seconds only 0.6c remains, while a taker needs about 2.2c to break even. Free
  throws (dead ball) are already priced before ESPN stamps them. Any lag exists only for a few seconds and cannot be used with
  ESPN data plus minute-level execution.
- **Volatility signal:** when the market and ESPN WP disagree more, the price moves more over the next 10 minutes. The average
  10-minute move rises from about 4.5-5c to about 8c. The pattern holds on both venues and every split. It predicts risk, not
  direction, and using it to gate quotes did not make market-making profitable.
- **Late-game possession and timeout cells (last 2:00, leader up 3-8, priced 0.80-0.97):** no cell clears costs on the tune
  split. The best is "leader has the ball, trailer has 2+ timeouts" at +1.3c (CI -2.7 to +4.9), so no paper-log entry was added.
- **Paper-trading spec (frozen):** two rules, scored only on games after Nov 12. Paper only: no orders, no wagers, no routine.
  - (1) Favourite hold: in the last 12:00, buy the favourite once it is priced 0.90-0.98, and hold to the end.
  - (2) Run fade: before 6:00 of Q4, after a side's price falls 20c or more in 3 minutes, buy it and hold.
  - Files: `_paper-spec.md` and `.json`, hash 473ce622c29e. The earlier 17-rule registry is unchanged.
  - Decision rule: call an edge only if the conservative CI is above 0 under both fees after at least 300 trades.
- **CI fix:** the day-clustered bootstrap was coded correctly but was overconfident when every forward trade won. Reported CIs now
  use the widest of the day bootstrap, the game bootstrap and the exact binomial. The pre-registered favourite-hold result in
  the forward test changes from looking significant (+3.8c, CI +0.4 to +6.5) to not significant (CI -2.9 to +6.4).
- **Timing caveat:** Polymarket's stored price history is about a minute behind Kalshi. An audit against Polymarket's public
  on-chain trades shows the history labels match the on-chain trade times (best offset 0 to -15 s). So the minute gap is in the
  venue or its settlement, not a mislabelled history. That makes the final run's 60 s shift questionable. It can let
  Polymarket-based rules see up to a minute into the future, which makes them look better, not worse. Still, every Polymarket
  rule loses. Two positives should be treated as artifacts:
  - The forward directional-maker result (+2.5c per fill; holdout -0.7c).
  - The 11-second gap-convergence figure.

Detailed tables, both fee scenarios, the trial registry and caveats follow. Everything is simulated, with no real trades.