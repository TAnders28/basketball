# In-play NBA swings — plain-language findings (interim through 8:40 PM PT)

**Verdict: PASS. Do not trade these rules.** After about 2,000 registered variants, the tested market prices mostly behave
like calibrated probabilities. Swings are common, but buying dips and selling rebounds is not an edge once you pay the
spread, fees, slippage and a minute of delay.

- **Tim's evenly-matched comeback idea:** it shows up in ESPN's win-probability model, not in the markets. When the pregame
  price is 45-55% and the leader reaches 75-90%, the trailer beats ESPN WP by 6.3 percentage points (95% CI +4.3 to +8.5),
  but its market win rate minus price is -2.0 points (Kalshi, CI -8.1 to +4.3) and -1.5 points (Polymarket, CI -7.4 to +4.8).
  Controls in mismatched games do not change the result. Buying the trailer loses about 4-5c per contract on the holdout.
- **Market-making:** a conservative real-print fill model loses 1.1c per fill on holdout, even at the cheaper maker fee.
  Pulling quotes after ESPN jumps and quoting only calm states reduces the loss but does not remove it. A model that fills
  every order when the market merely touches it looks profitable by 0.1-0.2c; that is a queue-position fantasy.
- **Sub-minute lag:** for 23,333 field goals, Kalshi has moved most of the way within 5-10 seconds of ESPN's timestamp.
  At 10 seconds only 0.6c remains on average; a taker needs about 2.2c just to cover costs. Free throws are already priced
  before ESPN stamps them. One-minute-latency lag strategies do not work.
- **Volatility signal:** larger market-vs-ESPN disagreement predicts larger moves over the next 10 minutes (mean abs change
  rises from about 5c at <3c disagreement to about 8c at >10c). It predicts risk, not direction or profit.
- **Paper spec:** two frozen rules only (favourite-hold 0.90-0.98, last 12 minutes; and a 20c-in-3-minutes run fade), scored
  only on games after Nov 12. Saved as `_paper-spec.md/.json` (hash 473ce622c29e). Nothing placed and no routine created.
- **CI audit:** the day bootstrap was coded correctly but was overconfident when every forward trade won. Exact binomial
  bounds erase the apparent significance of the pre-registered favourite-hold rule: +3.8c becomes CI -2.9c to +6.4c.
- **Important timing caveat:** Polymarket's prices-history lags Kalshi by roughly a minute; the final v3 run moves it back
  60s. A new on-chain trade audit is testing whether that shift is justified. Until settled, disregard every positive
  Polymarket-timed or cross-venue result (including the directional-maker forward result). The negative Kalshi results
  and PASS verdict do not rely on that shift.

Details, both fee scenarios, tables and caveats follow. Figures are simulated, not realised trades.