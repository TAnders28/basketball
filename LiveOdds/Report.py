"""Read-only report for one NBA date. Pregame Elo is not an in-play probability.

    python -m LiveOdds.Report --date 2026-11-08
    python -m LiveOdds.Report --date 2026-11-08 --json

The real-source sample command is `python -m LiveOdds.Smoke`. Unit tests call
BuildReport on labeled fixtures and do not hit the network.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from decimal import Decimal

from LiveOdds.Sources import ClassifyKalshiMarket, ClassifyPolymarketMarket, LoadBundle
from LiveOdds.Time import EASTERN, FormatEt, FormatUtc, ParseInstant, ResolveNaiveWallClock
from LiveOdds.Validate import (
    EvaluateGameState, EvaluatePolymarketBook, FeeStatus, GuardModels, OvertimeRule, Readiness,
)

_SOURCE_NOTES = [
    {"name": "polymarket_gamma",
     "docs": ["https://docs.polymarket.com/market-data/discover-markets",
              "https://gamma-api.polymarket.com/sports",
              "https://gamma-api.polymarket.com/markets/keyset"],
     "role": "Discover NBA moneyline events. Gamma prices are indicative only."},
    {"name": "polymarket_clob",
     "docs": ["https://docs.polymarket.com/api-reference/market-data/get-order-book",
              "https://docs.polymarket.com/api-reference/market-data/get-fee-rate",
              "https://docs.polymarket.com/trading/fees"],
     "role": "Order-book snapshot timestamp, bid/ask depth, and base_fee. Executable only after validation."},
    {"name": "kalshi",
     "docs": ["https://docs.kalshi.com/getting_started/quick_start_market_data",
              "https://docs.kalshi.com/api-reference/market/get-market-orderbook",
              "https://docs.kalshi.com/api-reference/market/get-markets"],
     "role": "Public KXNBAGAME books. No snapshot timestamp, so not executable. updated_time is non-trading metadata."},
    {"name": "nba_scoreboard",
     "docs": ["https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"],
     "documented_via": "nba_api.live.nba.endpoints.scoreboard.ScoreBoard",
     "role": "Game state only. Not a betting line. meta.time is used only when its zone can be resolved."},
]


def _Observed(wrapped):
    if not wrapped:
        return None
    return wrapped.get("observed_at")


def _ScoreboardIndex(body, game_date):
    board = (body or {}).get("scoreboard") or {}
    board_date = board.get("gameDate")
    if board_date != game_date:
        return board_date, {}, []
    index, seen = {}, []
    for game in board.get("games") or []:
        away = ((game.get("awayTeam") or {}).get("teamTricode") or "").upper()
        home = ((game.get("homeTeam") or {}).get("teamTricode") or "").upper()
        key = (away, home)
        if key in index:
            seen.append(key)
        index[key] = game
    return board_date, index, seen


def _KalshiLevels(rows):
    levels = []
    for row in rows or []:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return None
        try:
            price, size = Decimal(str(row[0])), Decimal(str(row[1]))
        except Exception:
            return None
        if price <= 0 or price >= 1 or size <= 0:
            return None
        levels.append({"price": str(row[0]), "size": str(row[1]), "decimal": price})
    return levels


def BuildKalshiQuote(markets, books) -> dict:
    """Indicative only. The documented order book has no snapshot timestamp."""
    contracts = []
    for market in markets:
        wrapped = (books or {}).get(market["ticker"]) or {}
        body = wrapped.get("body") or {}
        orderbook = body.get("orderbook_fp") if isinstance(body, dict) else None
        yes_levels = _KalshiLevels((orderbook or {}).get("yes_dollars")) if orderbook else None
        no_levels = _KalshiLevels((orderbook or {}).get("no_dollars")) if orderbook else None
        best = max(yes_levels, key=lambda row: row["decimal"]) if yes_levels else None
        summary_bid = market.get("yes_bid")
        touch_verified = False
        if best is not None and summary_bid not in (None, ""):
            touch_verified = Decimal(str(summary_bid)) == best["decimal"]
        derived = []
        if yes_levels is not None:
            for level in sorted(yes_levels, key=lambda row: row["decimal"], reverse=True):
                complement = Decimal("1") - level["decimal"]
                places = level["price"].split(".")
                quant = Decimal("1").scaleb(-(len(places[1]) if len(places) > 1 else 0))
                derived.append({"outcome": market["yes_sub_title"], "side": "ask_no",
                                "price": str(complement.quantize(quant)), "size": level["size"],
                                "ask_source": "kalshi_documented_complement",
                                "note": "A yes bid at price X is the no ask at 1-X. Docs return bids only."})
        meta_time, meta_kind = ParseInstant(market.get("metadata_updated_at"))
        contracts.append({
            "ticker": market["ticker"], "team": market["side"], "yes_sub_title": market["yes_sub_title"],
            "status": market.get("status"), "api_url": market["api_url"], "orderbook_url": wrapped.get("url") or market["orderbook_url"],
            "summary_yes_bid": summary_bid, "summary_yes_ask": market.get("yes_ask"),
            "summary_yes_bid_size": market.get("yes_bid_size"), "summary_yes_ask_size": market.get("yes_ask_size"),
            "summary_tag": "indicative_market_payload",
            "orderbook_yes_bids": None if yes_levels is None else [
                {"outcome": market["yes_sub_title"], "side": "bid_yes", "price": row["price"], "size": row["size"]}
                for row in sorted(yes_levels, key=lambda row: row["decimal"], reverse=True)[:5]],
            "orderbook_no_bids": None if no_levels is None else [
                {"side": "bid_no", "price": row["price"], "size": row["size"]}
                for row in sorted(no_levels, key=lambda row: row["decimal"], reverse=True)[:5]],
            "derived_no_asks": derived[:5],
            "touch_matches_summary": touch_verified,
            "metadata_updated_at": market.get("metadata_updated_at"),
            "metadata_timestamp_kind": meta_kind,
            "metadata_is_quote_timestamp": False,
            "metadata_note": "updated_time is the last non-trading metadata update, not an order-book snapshot time.",
            "quote_timestamp_utc": None,
            "observed_at_utc": FormatUtc(wrapped["observed_at"]) if wrapped.get("observed_at") else None,
            "book_error": body.get("error") if isinstance(body, dict) else "missing_book",
            "ignored_metadata_time_utc": FormatUtc(meta_time) if meta_time else None,
        })
    rules = "\n".join(market.get("rules") or "" for market in markets)
    settlement = OvertimeRule(rules)
    settlement["yes_notional_dollars"] = markets[0].get("notional_value_dollars") if markets else None
    settlement["yes_notional_source"] = "market.notional_value_dollars" if settlement["yes_notional_dollars"] else None
    return {
        "venue": "kalshi", "series_ticker": "KXNBAGAME", "executable": False, "indicative_only": True,
        "freshness": "unknown",
        "reason": "Get Market Orderbook has no snapshot timestamp. Retrieval time is not used as the quote time.",
        "fees": {"status": "unknown", "assumed_zero": False, "fee_adjusted_probability": None,
                 "note": "No fee field on the public market payload. Zero is not assumed.",
                 "docs": "https://docs.kalshi.com/api-reference/market/get-markets"},
        "settlement": settlement,
        "contracts": contracts,
        "event_ticker": markets[0].get("event_ticker") if markets else None,
    }


def _GroupKalshi(raw_markets, game_date):
    groups, rejected = defaultdict(list), []
    for raw in raw_markets or []:
        classified = ClassifyKalshiMarket(raw, game_date)
        if classified["status"] == "keep":
            groups[classified["market"]["event_ticker"]].append(classified["market"])
        elif classified["status"] == "reject":
            rejected.append(classified)
    ready, ambiguous = {}, []
    for event_ticker, rows in groups.items():
        teams = {(row["away"], row["home"]) for row in rows}
        sides = {row["side"] for row in rows}
        pair = next(iter(teams))
        if len(rows) != 2 or len(teams) != 1 or sides != set(pair):
            ambiguous.append({"event_ticker": event_ticker, "tickers": [row["ticker"] for row in rows]})
            continue
        ready[frozenset(pair)] = rows
    return ready, ambiguous, rejected


def _EmptyState(reason, observed_at):
    return {"source": "nba_cdn_scoreboard", "usable": False, "freshness": "unknown", "phase": "unknown",
            "blockers": [reason], "observed_at_utc": FormatUtc(observed_at) if observed_at else None,
            "timestamp_utc": None, "timestamp_et": None}


def BuildReport(bundle: dict) -> dict:
    """Assemble one date. `live_sample` is true only when the caller actually read public sources."""
    game_date = bundle["date"]
    models = GuardModels(bundle.get("pregame_model"))
    blockers = []
    if bundle.get("polymarket_error"):
        blockers.append("polymarket_" + str(bundle["polymarket_error"]))
    if bundle.get("scoreboard_error"):
        blockers.append("scoreboard_" + str(bundle["scoreboard_error"]))
    if bundle.get("kalshi_error"):
        blockers.append("kalshi_" + str(bundle["kalshi_error"]))

    kept, rejected = [], []
    by_pair = defaultdict(list)
    if not bundle.get("polymarket_error"):
        for raw in bundle.get("polymarket_markets") or []:
            classified = ClassifyPolymarketMarket(raw, game_date)
            if classified["status"] == "keep":
                by_pair[(classified["market"]["away"], classified["market"]["home"])].append(classified["market"])
            elif classified["status"] == "reject":
                rejected.append(classified)
    ambiguous_pairs = {pair for pair, rows in by_pair.items() if len(rows) != 1}
    for pair, rows in by_pair.items():
        if pair not in ambiguous_pairs:
            kept.append(rows[0])

    scoreboard = bundle.get("scoreboard") or {}
    board_body = scoreboard.get("body") if not bundle.get("scoreboard_error") else None
    board_observed = _Observed(scoreboard)
    board_date, index, ambiguous_board = _ScoreboardIndex(board_body, game_date)
    if board_body is not None and board_date != game_date:
        blockers.append("scoreboard_date_mismatch")
    resolved_time = {"freshness": "unknown", "utc": None, "raw": None, "timezone_resolution": "missing"}
    if board_body is not None and board_observed is not None:
        resolved_time = ResolveNaiveWallClock(((board_body.get("meta") or {}).get("time")), board_observed)

    kalshi_ready, kalshi_ambiguous, kalshi_rejected = _GroupKalshi(
        [] if bundle.get("kalshi_error") else bundle.get("kalshi_markets"), game_date)
    if kalshi_ambiguous:
        blockers.append("kalshi_ambiguous_event")

    games = []
    for market in kept:
        pair = (market["away"], market["home"])
        state_game = None
        if pair in ambiguous_board or pair[::-1] in ambiguous_board:
            game_state = _EmptyState("ambiguous_scoreboard_game", board_observed)
        elif board_body is None:
            game_state = _EmptyState("scoreboard_unavailable", board_observed)
        elif board_date != game_date:
            game_state = _EmptyState("scoreboard_date_mismatch", board_observed)
        elif pair in index:
            state_game = index[pair]
            game_state = EvaluateGameState(state_game, resolved_time, board_observed, board_date, game_date)
        elif (market["home"], market["away"]) in index:
            game_state = _EmptyState("home_away_mismatch", board_observed)
        else:
            game_state = _EmptyState("team_not_on_scoreboard", board_observed)
        settlement = OvertimeRule(market.get("description"))
        settlement["resolution_source"] = market.get("resolution_source")
        settlement["per_share_payout"] = "unknown"
        evaluated = dict(market)
        evaluated["includes_overtime"] = settlement["includes_overtime"]
        raw_phase = game_state.get("phase") or "unknown"
        if game_state.get("usable"):
            quote_phase = raw_phase
        elif raw_phase in ("final", "overtime"):
            quote_phase = raw_phase
        else:
            # An unresolved game clock must not open the looser pregame age window.
            quote_phase = "unknown"
        quote = EvaluatePolymarketBook(evaluated, bundle.get("books") or {}, quote_phase)
        fee_payloads = {}
        for token_id in market["token_ids"]:
            body = ((bundle.get("fees") or {}).get(token_id) or {}).get("body")
            if isinstance(body, dict) and "base_fee" in body:
                fee_payloads[token_id] = body
        fees = FeeStatus(market, fee_payloads, market["token_ids"])
        kalshi_rows = kalshi_ready.get(frozenset(pair))
        kalshi = BuildKalshiQuote(kalshi_rows, bundle.get("kalshi_books")) if kalshi_rows else None
        if kalshi is None and not bundle.get("kalshi_error"):
            kalshi = {"venue": "kalshi", "executable": False, "freshness": "unknown",
                      "reason": "no_matching_open_contract", "contracts": []}
        game_blockers = list(quote["blockers"])
        if not game_state.get("usable"):
            game_blockers.extend(game_state.get("blockers") or [])
        games.append({
            "matchup": f"{market['away']} @ {market['home']}",
            "away": market["away"], "home": market["home"], "date": game_date,
            "readiness_tier": Readiness(quote, game_state),
            "assessment_tier": "unavailable",
            "blockers": list(dict.fromkeys(game_blockers)),
            "identity": {"slug": market["slug"], "condition_id": market["condition_id"],
                         "question": market["question"], "outcomes": market["outcomes"], **market["urls"]},
            "quote": quote,
            "indicative_gamma": market["indicative"],
            "fees": fees,
            "settlement": settlement,
            "game_state": game_state,
            "venue_reported_state": {
                "source": "polymarket_event", "authoritative": False,
                "score_team_attribution": "unknown", **market["event"],
                "note": "The event score string is unlabeled and is not the game-state source.",
            },
            "kalshi": kalshi,
            "models": models,
        })

    for pair in sorted(ambiguous_pairs):
        slugs = [row["slug"] for row in by_pair[pair]]
        games.append({"matchup": f"{pair[0]} @ {pair[1]}", "away": pair[0], "home": pair[1], "date": game_date,
                      "readiness_tier": "blocked", "assessment_tier": "unavailable",
                      "blockers": ["ambiguous_event_mapping"], "identity": {"slugs": slugs},
                      "quote": {"executable": False, "freshness": "unknown", "levels": []},
                      "models": models})
    for item in rejected:
        games.append({"matchup": item.get("slug"), "date": game_date, "readiness_tier": "blocked",
                      "assessment_tier": "unavailable", "blockers": [item["reason"]],
                      "identity": {"slug": item.get("slug"), "signals": item.get("signals")},
                      "quote": {"executable": False, "indicative_only": True, "freshness": "unknown"},
                      "models": models})
    for item in kalshi_rejected:
        blockers.append("kalshi_" + item["reason"])

    rank = ["executable_quote", "executable_unaligned", "not_executable", "blocked"]
    present = {game["readiness_tier"] for game in games}
    if not games and blockers:
        reporting = "blocked"
    elif not games:
        reporting = "blocked"
        blockers.append("no_matching_market")
    else:
        reporting = next(tier for tier in rank if tier in present)
    observed_candidates = [board_observed]
    for wrapped in (bundle.get("books") or {}).values():
        observed_candidates.append(wrapped.get("observed_at"))
    observed = [moment for moment in observed_candidates if moment is not None]
    observed_at = max(observed) if observed else None
    return {
        "report": "live_odds",
        "live_sample": bool(bundle.get("live_sample")),
        "fixture": not bool(bundle.get("live_sample")),
        "date": game_date,
        "observed_at_utc": FormatUtc(observed_at) if observed_at else None,
        "observed_at_et": FormatEt(observed_at) if observed_at else None,
        "readiness_tier": reporting,
        "assessment_tier": "unavailable",
        "assessment_reason": models["assessment_reason"],
        "supported_edge": None,
        "expected_value": None,
        "market_implied_probability_is_model_forecast": False,
        "models": models,
        "sources": _SOURCE_NOTES,
        "blockers": list(dict.fromkeys(blockers)),
        "games": games,
        "counts": {tier: sum(game["readiness_tier"] == tier for game in games) for tier in rank},
    }


def FormatReport(report: dict) -> str:
    lines = [
        f"# NBA live odds {report['date']} observed {report.get('observed_at_et') or 'unknown'}",
        "",
        f"Reporting tier: {report['readiness_tier']}. Assessment tier: {report['assessment_tier']}.",
        report["assessment_reason"],
        "Supported edge: none. Expected value: not computed.",
        "",
    ]
    if report.get("fixture"):
        lines.append("This output is not a live sample.")
        lines.append("")
    if report["blockers"]:
        lines.append("Slate blockers: " + ", ".join(report["blockers"]))
        lines.append("")
    lines.append("| Game | Reporting | Quote time ET | Bid / ask | Game state ET | Phase | Assessment |")
    lines.append("|---|---|---|---|---|---|---|")
    for game in report["games"]:
        quote = game.get("quote") or {}
        touches = quote.get("touches") or []
        book = ", ".join(f"{row['abbr']} {row['bid']}-{row['ask']}" for row in touches) or "-"
        state = game.get("game_state") or {}
        lines.append(
            f"| {game.get('matchup')} | {game['readiness_tier']} | {quote.get('quote_timestamp_et') or quote.get('freshness') or '-'} | "
            f"{book} | {state.get('timestamp_et') or '-'} | {state.get('phase') or '-'} | unavailable |")
        if game.get("blockers"):
            lines.append(f"| | blockers | {', '.join(game['blockers'])} | | | | |")
    return "\n".join(lines)


def Main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Read-only NBA odds report for one Eastern date. Does not place orders.")
    parser.add_argument("--date", default=datetime.now(EASTERN).strftime("%Y-%m-%d"))
    parser.add_argument("--json", action="store_true", help="Print the full report JSON")
    parser.add_argument("--pregame-prob", type=float, default=None,
                        help="Optional pregame home probability. Labeled and ignored for edge.")
    args = parser.parse_args(argv)
    datetime.strptime(args.date, "%Y-%m-%d")
    pregame = None
    if args.pregame_prob is not None:
        pregame = {"name": "cli_supplied_pregame", "home_prob": args.pregame_prob}
    report = BuildReport(LoadBundle(args.date, pregame_model=pregame, live_sample=True))
    print(json.dumps(report, indent=2) if args.json else FormatReport(report))


if __name__ == "__main__":
    Main()
