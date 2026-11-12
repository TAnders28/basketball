# AGENTS.md — basketball

Shared conventions for agents working in this repo. Keep changes small and local unless Tim says otherwise.

## Layout

- `GetSeasonGames.py` — season pull (stats.nba.com via nba_api); writes `data/season/games_<year>.json` etc.
- `Backtest/` — leak-free pregame backtest, PredictDay, grading, AssessLive entrypoints
- `LiveOdds/` — read-only current quotes (Polymarket / Kalshi / NBA CDN). No orders.
- `PlayByPlay/` — placeholder
- `tests/` — unittest (fixtures; not live readiness)
- `data/` — local only (gitignored). Season pulls, backtest reports, prediction logs.
- `docs/` — writeups (e.g. `docs/live_odds.md`)
- `examples/` — small tracked samples

## Entry points

```text
.venv\Scripts\python.exe GetSeasonGames.py --season 2024
.venv\Scripts\python.exe -m Backtest.RunBacktest --season 2024
.venv\Scripts\python.exe -m Backtest.PredictDay --date YYYY-MM-DD
.venv\Scripts\python.exe -m Backtest.GradePredictions --season 2026
.venv\Scripts\python.exe -m Backtest.AssessLive --date YYYY-MM-DD
.venv\Scripts\python.exe -m LiveOdds.Report --date YYYY-MM-DD
.venv\Scripts\python.exe -m Matchups.PullData --seasons 2026 2025 2024 2023 --prior-seasons 2022   # resumable, polite
.venv\Scripts\python.exe -m Matchups.RunAll --workers 10    # research grid + sim; caps at floor(cores/2), BelowNormal
.venv\Scripts\python.exe -m Matchups.Report --out-dir <AgentDocs folder>
.venv\Scripts\python.exe -m unittest discover -s tests
```

## Conventions

- Functions / classes / module files: PascalCase. Variables: snake_case.
- Code and docs in git; data stays local (`/data/`, `*.json`, `*.jsonl`, `*.parquet`, `.venv/`).
- No look-ahead: backtests walk day-by-day; picks on day D only see games from earlier days. PredictDay freezes completed games with `game_date < target`.
- Without moneylines: report pick **accuracy** only — no invented P&L.
- Working fee estimate for edge math: **ask + 0.02** per $1 contract (Tim's estimate; not a verified exchange fee). LiveOdds may still leave fee fields unreconciled.
- Live AssessLive: with no validated in-play model, action is always **PASS** (`no live model`). Stale or misaligned quotes also **PASS**. Default freshness: quote 30s, game state 45s (CLI-overridable).
- No cloud coding agents for this checkout unless Tim asks. **Push only when Tim explicitly says OK.** Local commits are fine when authorized.
- Do not place orders or wagers from this repo.
