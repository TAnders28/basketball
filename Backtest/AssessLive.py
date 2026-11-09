r"""One-shot live assessment writer (no polling, no orders).

    .venv\Scripts\python.exe -m Backtest.AssessLive --date 2026-11-08

Runs LiveOdds.Report once, appends one jsonl row per market side to
E:\AgentDocs\Basketbot\<date>_live\<date>_live.jsonl, and overwrites
<date>_live_latest.md.

Default freshness matches LiveOdds: quote 30s, game state 45s (CLI-overridable).
With no validated in-play model, action is always PASS ("no live model").
model_prob and edge are always null here.
"""

import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from Backtest.Odds import FEE_BASIS
from LiveOdds.Report import BuildReport, FormatReport
from LiveOdds.Sources import LoadBundle
from LiveOdds.Time import (
    MAX_GAME_STATE_AGE_SECONDS,
    MAX_LIVE_QUOTE_AGE_SECONDS,
    AgeSeconds,
    EASTERN,
    FormatEt,
    Freshness,
    ParseInstant,
)

DEFAULT_OUT_ROOT = os.path.join("E:\\", "AgentDocs", "Basketbot")


def OutDir(game_date: str, root: str = None) -> str:
    base = root or DEFAULT_OUT_ROOT
    return os.path.join(base, f"{game_date}_live")


def JsonlPath(game_date: str, root: str = None) -> str:
    return os.path.join(OutDir(game_date, root), f"{game_date}_live.jsonl")


def LatestMdPath(game_date: str, root: str = None) -> str:
    return os.path.join(OutDir(game_date, root), f"{game_date}_live_latest.md")


def _ScoreString(state: dict) -> str:
    if not state:
        return None
    away = state.get("away_score")
    home = state.get("home_score")
    if away is None and home is None:
        # Some payloads nest scores
        away = (state.get("away") or {}).get("score") if isinstance(state.get("away"), dict) else away
        home = (state.get("home") or {}).get("score") if isinstance(state.get("home"), dict) else home
    if away is None and home is None:
        return state.get("score") or state.get("score_text")
    return f"{away}-{home}"


def Requalify(quote: dict, state: dict, readiness_tier: str,
              max_quote_age: float, max_state_age: float, observed_at) -> dict:
    """Apply CLI freshness overrides on top of LiveOdds readiness."""
    observed = ParseInstant(observed_at)[0] if observed_at else None
    if observed is None and quote.get("observed_at_utc"):
        observed = ParseInstant(quote.get("observed_at_utc"))[0]

    quote_time = ParseInstant(quote.get("quote_timestamp_utc"))[0]
    state_time = ParseInstant((state or {}).get("timestamp_utc"))[0]

    quote_freshness = Freshness(quote_time, observed, max_quote_age) if observed else (quote.get("freshness") or "unknown")
    state_freshness = Freshness(state_time, observed, max_state_age) if observed else ((state or {}).get("freshness") or "unknown")

    executable = bool(quote.get("executable"))
    aligned = readiness_tier == "executable_quote"
    fresh_ok = quote_freshness == "fresh" and (
        state_freshness == "fresh" or (state or {}).get("phase") == "scheduled"
    )
    # Scheduled pregame may use looser quote window already inside LiveOdds; for overrides,
    # require quote fresh; state may be unknown when scheduled and unused.
    if (state or {}).get("phase") == "scheduled" and (state or {}).get("usable"):
        fresh_ok = quote_freshness == "fresh"
    elif not (state or {}).get("usable"):
        fresh_ok = False

    quote_qualified = bool(executable and fresh_ok and aligned)
    stale = quote_freshness == "stale" or state_freshness == "stale" or (
        readiness_tier == "executable_unaligned" and "quote_state" in json.dumps((state or {}).get("blockers") or [])
    )
    return {
        "quote_qualified": quote_qualified,
        "quote_freshness": quote_freshness,
        "state_freshness": state_freshness,
        "stale": stale,
        "quote_age_seconds": AgeSeconds(quote_time, observed) if observed else None,
        "state_age_seconds": AgeSeconds(state_time, observed) if observed else None,
    }


def RowsFromReport(report: dict, max_quote_age: float, max_state_age: float,
                   observed_at_override: str = None) -> list:
    observed_at = observed_at_override or report.get("observed_at_et") or FormatEt(datetime.now(EASTERN))
    observed_utc = report.get("observed_at_utc")
    rows = []
    for game in report.get("games") or []:
        quote = game.get("quote") or {}
        state = game.get("game_state") or {}
        readiness = game.get("readiness_tier") or "blocked"
        qual = Requalify(quote, state, readiness, max_quote_age, max_state_age, observed_utc)
        venue = "polymarket"
        market = (game.get("identity") or {}).get("slug") or game.get("matchup")
        touches = quote.get("touches") or []
        if not touches:
            rows.append({
                "observed_at": observed_at,
                "game_id": state.get("game_id") or (game.get("identity") or {}).get("condition_id"),
                "teams": game.get("matchup"),
                "score": _ScoreString(state),
                "period": state.get("period") or state.get("period_text"),
                "clock": state.get("clock") or state.get("game_clock"),
                "state_time": state.get("timestamp_et") or state.get("timestamp_utc"),
                "venue": venue,
                "market": market,
                "side": None,
                "bid": None,
                "ask": None,
                "size": None,
                "price_time": quote.get("quote_timestamp_et") or quote.get("quote_timestamp_utc"),
                "model_prob": None,
                "edge": None,
                "action": "PASS",
                "reason": "no live model",
                "fee_basis": FEE_BASIS,
                "quote_qualified": False,
                "readiness_tier": readiness,
                "executable": bool(quote.get("executable")),
            })
            continue
        for touch in touches:
            size = touch.get("ask_size")
            if size is None:
                size = touch.get("size")
            rows.append({
                "observed_at": observed_at,
                "game_id": state.get("game_id") or (game.get("identity") or {}).get("condition_id"),
                "teams": game.get("matchup"),
                "score": _ScoreString(state),
                "period": state.get("period") or state.get("period_text"),
                "clock": state.get("clock") or state.get("game_clock"),
                "state_time": state.get("timestamp_et") or state.get("timestamp_utc"),
                "venue": venue,
                "market": market,
                "side": touch.get("abbr") or touch.get("outcome") or touch.get("side"),
                "bid": touch.get("bid"),
                "ask": touch.get("ask"),
                "size": size,
                "price_time": quote.get("quote_timestamp_et") or quote.get("quote_timestamp_utc"),
                "model_prob": None,
                "edge": None,
                "action": "PASS",
                "reason": "no live model",
                "fee_basis": FEE_BASIS,
                "quote_qualified": qual["quote_qualified"],
                "readiness_tier": readiness,
                "executable": bool(quote.get("executable")),
                "quote_freshness": qual["quote_freshness"],
                "state_freshness": qual["state_freshness"],
            })
    return rows


def AppendRows(path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def WriteLatestMarkdown(path: str, report: dict, rows: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        FormatReport(report),
        "",
        "## AssessLive",
        "",
        f"- rows appended: {len(rows)}",
        f"- fee_basis: {FEE_BASIS}",
        "- action: always PASS (no validated in-play model)",
        f"- quote_qualified true count: {sum(1 for r in rows if r.get('quote_qualified'))}",
        "",
        "| Teams | Side | Bid | Ask | Qualified | Action | Reason |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('teams')} | {row.get('side') or '-'} | {row.get('bid') or '-'} | "
            f"{row.get('ask') or '-'} | {row.get('quote_qualified')} | {row.get('action')} | {row.get('reason')} |"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def Assess(game_date: str, out_root: str = None, max_quote_age: float = None,
           max_state_age: float = None, report: dict = None) -> dict:
    max_quote_age = MAX_LIVE_QUOTE_AGE_SECONDS if max_quote_age is None else float(max_quote_age)
    max_state_age = MAX_GAME_STATE_AGE_SECONDS if max_state_age is None else float(max_state_age)
    if report is None:
        report = BuildReport(LoadBundle(game_date, live_sample=True))
    rows = RowsFromReport(report, max_quote_age, max_state_age)
    jsonl = JsonlPath(game_date, out_root)
    latest = LatestMdPath(game_date, out_root)
    AppendRows(jsonl, rows)
    WriteLatestMarkdown(latest, report, rows)
    return {"date": game_date, "rows": len(rows), "jsonl": jsonl, "latest_md": latest,
            "max_quote_age": max_quote_age, "max_state_age": max_state_age,
            "report_tier": report.get("readiness_tier"), "assessment_tier": report.get("assessment_tier")}


def Main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="One-shot live assess writer (PASS without live model)")
    parser.add_argument("--date", default=datetime.now(EASTERN).strftime("%Y-%m-%d"))
    parser.add_argument("--out-root", default=None, help="Override AgentDocs root (tests)")
    parser.add_argument("--max-quote-age", type=float, default=None,
                        help=f"Seconds; default {MAX_LIVE_QUOTE_AGE_SECONDS}")
    parser.add_argument("--max-state-age", type=float, default=None,
                        help=f"Seconds; default {MAX_GAME_STATE_AGE_SECONDS}")
    args = parser.parse_args(argv)
    datetime.strptime(args.date, "%Y-%m-%d")
    result = Assess(args.date, out_root=args.out_root,
                    max_quote_age=args.max_quote_age, max_state_age=args.max_state_age)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    Main()

