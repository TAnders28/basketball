# In-play NBA swings - plain-language findings (final, Nov 12, 2026, about 9:40 PM PT; Round 7 added)

**Verdict: PASS. Do not trade any of these rules.** We registered about 3,900 trials (about 2,000 strategy variants, plus Round 7 reruns of every Polymarket rule under three timings). Each was tuned on 2024-25,
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
- **Timing (resolved in Round 7):** Polymarket's history labels match its on-chain trades, and its price trails ESPN by one minute
  in 94% of games. Rerunning every Polymarket-based rule with no shift (and with per-game offsets) leaves every rule losing. The
  one forward positive in the study, the directional maker off the Polymarket gap (+2.5c per fill), becomes -1.2c (CI -1.7 to
  -0.6) without the shift. It came from reading Polymarket a minute ahead.
- **Round 7 (added 9:00-9:40 PM):**
  - Queue position: quoting Kalshi earns +0.2c per fill on the holdout only if we are first in line. With 50-250 contracts ahead
    of us it is zero to negative, and -1.2 to -1.4c if we only count prints through our price. Not tradable.
  - Moneyline vs spread markets on Kalshi (170 forward games): they agree within 1.6c (median), no arbitrage after fees, and
    neither leads the other at one-minute resolution. Trading the gap lost 3.2c per contract (CI -5.6 to -1.3).
  - Stars who leave and do not return: only 20 cases. In the holdout the price kept sliding against the team (-5.9c over 5
    minutes) but not in the forward games, and fading the team lost money. ESPN data has no injury flag or confirmation time,
    so this needs news timestamps to test properly.

Detailed tables, both fee scenarios, the trial registry and caveats follow. Everything is simulated, with no real trades.