# Live odds report

Read-only current quotes for one NBA date. This does not place orders, size bets, or
produce an in-play win probability. `Backtest.PredictDay` is unchanged and remains
the pregame Elo / record / rest report.

```text
python -m LiveOdds.Report --date 2026-11-08
python -m LiveOdds.Report --date 2026-11-08 --json
python -m LiveOdds.Smoke
python -m LiveOdds.Smoke --date 2026-11-08 --table
python -m unittest tests.test_live_odds
```

`Report` and `Smoke` call the public endpoints below. `Smoke` is the real-source
command. Unit tests use labeled fixtures (`EXAMPLE_NOT_A_LIVE_QUOTE`) and do not
establish live readiness. A historical fixture is not a current quote.

The default date is the Eastern calendar date at process start. Instants are stored
in UTC and shown in Eastern. Retrieval time is recorded as `observed_at` and is
never copied into a missing source timestamp. If the source did not provide a
quote time, freshness is `unknown`.

## Sources

| Source | What it is used for | Docs |
|---|---|---|
| Polymarket Gamma | NBA moneyline discovery (`GET /sports`, then `GET /markets/keyset` with the NBA `primaryTagId` and `sports_market_types=moneyline`) | https://docs.polymarket.com/market-data/discover-markets |
| Polymarket CLOB | Bid, ask, size, and the order-book snapshot `timestamp` | https://docs.polymarket.com/api-reference/market-data/get-order-book |
| Polymarket CLOB fee | `base_fee` in basis points for each token | https://docs.polymarket.com/api-reference/market-data/get-fee-rate and https://docs.polymarket.com/trading/fees |
| Kalshi | Public `series_ticker=KXNBAGAME` markets and order books | https://docs.kalshi.com/getting_started/quick_start_market_data and https://docs.kalshi.com/api-reference/market/get-market-orderbook |
| NBA scoreboard | Which game is which. `meta.time` has no offset and is not an as-of. Same URL as `nba_api.live.nba.endpoints.scoreboard` | `https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json` |
| NBA play-by-play | Score as-of from action `timeActual` when that timestamp has an offset. Same family as `nba_api.live.nba.endpoints.playbyplay` | `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gameId}.json` |

Each kept Polymarket contract records:

- event page `https://polymarket.com/event/{slug}`
- Gamma event and market URLs
- `conditionId` and both CLOB token ids
- book URL `https://clob.polymarket.com/book?token_id={id}`

Kalshi records the market URL and the order-book URL. Its order book returns bids
only. A no ask at `1 - price` is labeled `kalshi_documented_complement` because
that equivalence is in the order-book docs. The book response has no snapshot
timestamp. `updated_time` is the last non-trading metadata update, so it is not a
quote time. Kalshi rows stay `freshness: unknown` and `executable: false`.

Gamma `bestBid`, `bestAsk`, and `outcomePrices` are tagged `indicative`. They are
not an executable book. An aggregate price with no contract id (for example an
ESPN consensus number) is rejected the same way.

## Fail closed

A Polymarket book is executable only when all of these hold:

- slug, `gameStartTime` (Eastern date), and `eventDate` name the requested date
- both outcome names resolve to the slug's two teams
- one moneyline for that pair (two matches are `ambiguous_event_mapping`)
- `acceptingOrders` is true, the market is not closed, and resolution has not started
- both tokens have a bid and an ask inside `(0, 1)`, size at least `min_order_size`
- best bid is below best ask on each token, and the two bids sum to at most 1 while the two asks sum to at least 1
- the two book timestamps parse and are within 2 seconds
- the earlier timestamp is not more than 5 seconds ahead of the observation
- age is at most 30 seconds, or 180 seconds only when a verified game state says the game is still scheduled
- the game is not final
- in overtime, the contract text explicitly includes overtime

Otherwise the row is `not_executable`, `executable_unaligned`, or `blocked`.
Suspended, closed, crossed, empty, stale, future, undated, and mismatched rows
do not become quotes. A source that does not return JSON with HTTP 200 is an
outage. The client is GET-only and allow-lists the four hosts above.

## Fees and settlement

Polymarket documents two different fee objects, and this report does not turn them into one rate.

- The 31 Mar 2026 changelog says fees are calculated from the market `feeSchedule` ([changelog](https://docs.polymarket.com/changelog/predictions)). `feeSchedule.rate` is the base rate in that calculation, `exponent` is the price-curve exponent, and `takerOnly` means only the taker pays ([market details](https://docs.polymarket.com/market-data/market-details)). The published formula is `fee = C × feeRate × p × (1 - p)` ([fees](https://docs.polymarket.com/trading/fees)). The sports category row there is taker `0.05`, maker `0`, rebate 15%.
- `GET /fee-rate` documents `base_fee` as basis points ([fee rate](https://docs.polymarket.com/api-reference/market-data/get-fee-rate)). `GET /clob-markets/{condition_id}` documents both `fd` (rate, exponent, taker-only) and `mbf` / `tbf` as maker and taker base fees in basis points ([CLOB market info](https://docs.polymarket.com/api-reference/markets/get-clob-market-info)).

No page read for this change says `base_fee` equals `feeSchedule.rate × 10000`. That conversion is not applied. When both a schedule rate and a basis-point fee are present, status is `unreconciled`, `single_fee_rate` is null, and no fee-adjusted price is computed. `explicit_zero` is used only when every present schedule rate and basis-point field is explicitly 0. A missing field is not filled with 0.

After resolution, a winning outcome token redeems for 1 pUSD and a losing token for 0 ([resolution](https://docs.polymarket.com/concepts/resolution), [Polymarket 101](https://docs.polymarket.com/polymarket-101)). That redemption does not decide overtime. Overtime is true only when the contract text says the result includes overtime, and false only when it says overtime is excluded. Silence is `unknown`. During overtime, unknown or excluded settlement blocks the executable flag.

Kalshi `GET /series/KXNBAGAME` supplies `fee_type` and `fee_multiplier` ([get series](https://docs.kalshi.com/api-reference/market/get-series)). The dollar formula lives in the fee PDF and is not applied here. The series contract PDF is not parsed, so it does not decide overtime. Market `notional_value_dollars` is copied when present. `updated_time` remains non-trading metadata, not a quote time.

## Game state

The scoreboard `meta.time` string has no offset. It stays unused. It is not
interpreted as UTC or Eastern, including when a 12-hour shift would land near
the observation. Box-score tip fields such as `gameTimeUTC` are the scheduled
start, not the score as-of.

When play-by-play is read, the as-of is the latest action `timeActual` that
parses with an explicit offset (`nba_api.live.nba.endpoints.playbyplay`). The
score on that same action is the score. A naive `timeActual` is skipped. The
quote snapshot and that play must be within 30 seconds, and the play must be
fresh against the observation (45 seconds). Those windows are this report's
fail-closed limits, not an NBA service level. If they are missed, the book can
still be executable and the tier stays `executable_unaligned`.

The Polymarket event `score` string does not name which team has which total,
so it is not the game-state source.

## Tiers

`readiness_tier` is only about the quote report:

- `executable_quote`: verified book and a fresh game-state timestamp with an explicit offset
- `executable_unaligned`: verified book, game state not usable (missing, stale, or unknown clock)
- `not_executable`: a mapped market whose book or contract failed a check
- `blocked`: outage, or no usable mapping

`assessment_tier` is `unavailable` on every report. `Backtest` Elo, record, and
rest strategies are pregame only. This repository has no historical in-play
state-and-odds validation, and a calibrated live probability is not produced
from those backtests. `historical_state_odds_validation` is false and
`calibrated_live_probability` is null. Pregame Elo from `Backtest.PredictDay`
is labeled `pregame_only` when someone passes it in, and `usable_as_inplay`
stays false. `supported_edge` and `expected_value` stay null. A market bid or
ask is tagged `market_price_not_a_model_forecast`.
