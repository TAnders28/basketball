# basketball

NBA statistic retrieval and strategy backtesting. Same layout as `football` / `baseball`.

## Layout

- `GetSeasonGames.py` - pull NBA seasons (stats.nba.com via `nba_api`, no key) into readable JSON/JSONL
- `data/` - raw pulls and backtest reports (gitignored)
- `stats/` - derived season aggregates (gitignored)
- `Backtest/` - leak-free strategy backtesting
  - `Engine.py` - walks games day by day; picks only see earlier days
  - `Features.py` - season-to-date record, last-10, margin, rest days
  - `Model.py` - Elo model (home court, margin of victory, prior-season carryover)
  - `Strategies.py` - `AlwaysHome`, `BetterRecord`, `RestAdvantage`, `EloPick`, `ModelEdge`
  - `Odds.py` - American odds helpers (used once games carry moneylines)
  - `RunBacktest.py` - CLI runner + report
- `PlayByPlay/` - play-by-play (placeholder)
- `examples/` - small tracked sample outputs (`games_sample.json`)
- `docs/` - notes and writeups

## Setup

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

## Pull stats

A season is named by its start year (2024 = 2024-25).

    .venv\Scripts\python.exe GetSeasonGames.py --season 2024 --dry-run --limit 10   # quick check, writes nothing
    .venv\Scripts\python.exe GetSeasonGames.py --season 2023,2024                   # regular season + playoffs

Writes `data/season/games_<season>.json`, `team_games_<season>.jsonl`, `pull_summary_<season>.json`.

## Backtest

    .venv\Scripts\python.exe -m Backtest.RunBacktest --games-file examples\games_sample.json   # smoke test
    .venv\Scripts\python.exe -m Backtest.RunBacktest --season 2024                            # Elo seeded from 2023 if pulled

Reports go to `data/backtest/reports/` (`REPORT.md`, `report_<season>.json`, `picks_<season>_<strategy>.csv`).

Free NBA game logs have no betting lines, so strategies are scored on pick accuracy. If a game record
has `home_moneyline` / `away_moneyline`, the engine also runs a $10,000 flat-$200-stake bankroll.

## Conventions

- Functions/classes PascalCase, variables snake_case.
- Code only in git; data stays local.
