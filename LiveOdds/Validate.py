"""Fail-closed checks for books, fees, game state, and the pregame/in-play split.

A market price is never a model forecast. Pregame Elo is never an in-play probability.
Nothing here computes expected value.
"""

from decimal import Decimal

from LiveOdds.Time import (
    MAX_BOOK_TIMESTAMP_GAP_SECONDS, MAX_GAME_STATE_AGE_SECONDS, MAX_QUOTE_STATE_SKEW_SECONDS,
    AgeSeconds, FormatEt, FormatUtc, Freshness, GapSeconds, ParseInstant, QuoteMaxAge,
)

ONE = Decimal("1")
ZERO = Decimal("0")


def _Decimal(value):
    return Decimal(str(value))


def _Text(value) -> str:
    return "" if value is None else str(value)


def OvertimeRule(text) -> dict:
    """Read settlement text. Only explicit overtime language counts; silence is unknown."""
    raw = _Text(text)
    lowered = raw.lower()
    includes = None
    evidence = None
    if "including any overtime" in lowered or "including overtime" in lowered:
        includes = True
        evidence = _Sentence(raw, "overtime")
    elif any(phrase in lowered for phrase in (
            "excluding overtime", "exclude overtime", "excludes overtime",
            "regulation only", "does not include overtime", "not including overtime")):
        includes = False
        evidence = _Sentence(raw, "overtime") or _Sentence(raw, "regulation")
    return {"includes_overtime": includes, "evidence": evidence,
            "status": "sourced" if includes is not None else "unknown"}


def _Sentence(text: str, needle: str):
    for part in text.replace("\r", "").split("\n"):
        if needle in part.lower() and part.strip():
            return part.strip()
    return None


def FeeStatus(market: dict, clob_fees: dict, token_ids=None) -> dict:
    """Record fee fields under the roles the docs give them. Do not pick one number.

    The 31 Mar 2026 changelog says to calculate fees from `feeSchedule`.
    `GET /fee-rate` documents `base_fee` as basis points. `GET /clob-markets/{condition_id}`
    documents both `fd` (rate, exponent, taker-only) and `mbf`/`tbf` (basis points).
    Nothing we read says `base_fee == feeSchedule.rate × 10000`, so that conversion is not applied.
    """
    schedule = market.get("fee_schedule")
    rate = None
    if isinstance(schedule, dict) and schedule.get("rate") is not None:
        rate = _Decimal(schedule["rate"])
    base_fees = {}
    for token_id, payload in (clob_fees or {}).items():
        if isinstance(payload, dict) and payload.get("base_fee") is not None:
            base_fees[str(token_id)] = int(payload["base_fee"])
    maker = market.get("maker_base_fee")
    taker = market.get("taker_base_fee")
    fees_enabled = market.get("fees_enabled")
    taker_only = schedule.get("taker_only") if isinstance(schedule, dict) else None
    required = [str(token_id) for token_id in (token_ids or [])]
    missing = [token_id for token_id in required if token_id not in base_fees]
    maker_unreconciled = taker_only is True and maker not in (None, 0)
    explicit_zero = (rate == 0 and base_fees and all(bps == 0 for bps in base_fees.values())
                     and not missing and maker in (None, 0) and taker in (None, 0)
                     and fees_enabled is False)
    if explicit_zero:
        status = "explicit_zero"
    elif rate is not None and base_fees:
        status = "unreconciled"
    elif rate is not None or base_fees:
        status = "partial"
    else:
        status = "unknown"
    return {
        "status": status,
        "assumed_zero": False,
        "source_says_zero": status == "explicit_zero",
        "reconciled": False,
        "single_fee_rate": None,
        "documented_equivalence": None,
        "maker_fields_unreconciled": maker_unreconciled,
        "clob_fee_tokens_missing": missing,
        "fee_adjusted_probability": None,
        "fee_adjusted_reason": (
            "No single fee rate is selected. feeSchedule is the documented calculation object, "
            "and CLOB base_fee is a separate basis-point field. A fee-adjusted price is not computed."),
        "field_roles": {
            "feeSchedule": "calculation source per the 31 Mar 2026 changelog; rate is the base rate in the published formula",
            "clob_base_fee": "basis points on GET /fee-rate; not converted into feeSchedule.rate",
            "makerBaseFee_takerBaseFee": "recorded as returned; not treated as the feeSchedule rate",
        },
        "gamma_fees_enabled": fees_enabled,
        "gamma_fee_type": market.get("fee_type"),
        "gamma_fee_schedule": schedule,
        "gamma_maker_base_fee": maker,
        "gamma_taker_base_fee": taker,
        "clob_base_fee_bps": base_fees or None,
        "sources": [
            "https://docs.polymarket.com/changelog/predictions",
            "https://docs.polymarket.com/market-data/market-details",
            "https://docs.polymarket.com/trading/fees",
            "https://docs.polymarket.com/api-reference/market-data/get-fee-rate",
            "https://docs.polymarket.com/api-reference/markets/get-clob-market-info",
        ],
    }


def _Levels(raw_levels) -> tuple:
    """Return (levels, error). Prices must be inside (0, 1) and sizes positive."""
    levels = []
    for row in raw_levels or []:
        if not isinstance(row, dict) or "price" not in row or "size" not in row:
            return [], "level_missing_price_or_size"
        try:
            price = _Decimal(row["price"])
            size = _Decimal(row["size"])
        except Exception:
            return [], "level_not_numeric"
        if price <= ZERO or price >= ONE or size <= ZERO:
            return [], "level_out_of_range"
        levels.append({"price": price, "size": size, "price_text": str(row["price"]), "size_text": str(row["size"])})
    return levels, None


def _Best(levels, want_max: bool):
    if not levels:
        return None
    return max(levels, key=lambda row: row["price"]) if want_max else min(levels, key=lambda row: row["price"])


def _Touch(levels, want_max: bool, minimum: Decimal):
    eligible = [row for row in levels if row["size"] >= minimum]
    return _Best(eligible, want_max)


def _PublicLevel(row, outcome, side, executable_touch: bool) -> dict:
    return {"outcome": outcome, "side": side, "price": row["price_text"], "size": row["size_text"],
            "executable_touch": executable_touch}


def EvaluatePolymarketBook(market: dict, token_books: dict, phase: str) -> dict:
    """Verify a two-outcome CLOB book. Gamma summaries are indicative only.

    token_books maps token id -> {body, observed_at, url}.
    """
    blockers = []
    outcomes = market.get("outcomes") or []
    if len(outcomes) != 2:
        blockers.append("outcome_count")
    if market.get("closed") is True or market.get("accepting_orders") is not True:
        blockers.append("suspended_or_closed")
    if market.get("enable_order_book") is not True:
        blockers.append("order_book_disabled")
    if market.get("archived") is True:
        blockers.append("archived")
    resolution = market.get("resolution_status")
    if resolution not in (None, ""):
        blockers.append("resolution_" + str(resolution))
    if phase == "final":
        blockers.append("game_final")
    if phase == "overtime" and market.get("includes_overtime") is not True:
        blockers.append("overtime_settlement_" + ("unknown" if market.get("includes_overtime") is None else "excludes_ot"))

    parsed = []
    observed_times = []
    for outcome in outcomes:
        token_id = str(outcome.get("token_id"))
        wrapped = (token_books or {}).get(token_id)
        if not wrapped or not wrapped.get("body"):
            blockers.append("missing_book")
            continue
        body = wrapped["body"]
        if body.get("error"):
            blockers.append("book_error")
            continue
        if str(body.get("asset_id")) != token_id:
            blockers.append("token_mismatch")
        if market.get("condition_id") and str(body.get("market", "")).lower() != str(market["condition_id"]).lower():
            blockers.append("condition_mismatch")
        source_time, kind = ParseInstant(body.get("timestamp"))
        bids, bid_error = _Levels(body.get("bids"))
        asks, ask_error = _Levels(body.get("asks"))
        if bid_error or ask_error:
            blockers.append(bid_error or ask_error)
        minimum = body.get("min_order_size")
        try:
            minimum = _Decimal(minimum) if minimum not in (None, "") else None
        except Exception:
            minimum = None
            blockers.append("min_order_size_invalid")
        if minimum is None:
            blockers.append("min_order_size_missing")
        parsed.append({
            "outcome": outcome.get("name"), "abbr": outcome.get("abbr"), "side": outcome.get("side"),
            "token_id": token_id, "bids": bids, "asks": asks, "minimum": minimum,
            "source_time": source_time, "timestamp_kind": kind, "raw_timestamp": body.get("timestamp"),
            "observed_at": wrapped.get("observed_at"), "book_url": wrapped.get("url"),
            "tick_size": body.get("tick_size"), "hash": body.get("hash"),
        })
        if wrapped.get("observed_at") is not None:
            observed_times.append(wrapped["observed_at"])

    if len(parsed) != 2 or any(row["observed_at"] is None for row in parsed):
        blockers.append("missing_observation")
    observed_at = max(observed_times) if observed_times else None
    quote_time = None
    if len(parsed) == 2 and all(row["source_time"] is not None for row in parsed):
        quote_time = min(row["source_time"] for row in parsed)
    if len(parsed) == 2:
        if any(row["timestamp_kind"] in ("missing", "naive", "invalid") or row["source_time"] is None for row in parsed):
            blockers.append("quote_timestamp_" + ",".join(sorted({row["timestamp_kind"] for row in parsed})))
        gap = GapSeconds(parsed[0]["source_time"], parsed[1]["source_time"])
        if gap is None or gap > MAX_BOOK_TIMESTAMP_GAP_SECONDS:
            blockers.append("book_timestamp_mismatch")
        ticks = {str(row["tick_size"]) for row in parsed}
        if len(ticks) != 1 or None in ticks or "None" in ticks:
            blockers.append("tick_mismatch")
        for row in parsed:
            best_bid = _Best(row["bids"], True)
            best_ask = _Best(row["asks"], False)
            if best_bid is None or best_ask is None:
                blockers.append("missing_liquidity")
            elif best_bid["price"] >= best_ask["price"]:
                blockers.append("crossed_book")
            if row["minimum"] is not None:
                if _Touch(row["bids"], True, row["minimum"]) is None or _Touch(row["asks"], False, row["minimum"]) is None:
                    blockers.append("depth_below_min_order")
        if all(row["bids"] and row["asks"] for row in parsed):
            bid_sum = _Best(parsed[0]["bids"], True)["price"] + _Best(parsed[1]["bids"], True)["price"]
            ask_sum = _Best(parsed[0]["asks"], False)["price"] + _Best(parsed[1]["asks"], False)["price"]
            if bid_sum > ONE or ask_sum < ONE:
                blockers.append("crossed_complement")

    freshness = "unknown"
    if observed_at is not None and quote_time is not None and "quote_timestamp_" not in ",".join(blockers) and "book_timestamp_mismatch" not in blockers:
        freshness = Freshness(quote_time, observed_at, QuoteMaxAge(phase))
        if freshness != "fresh":
            blockers.append(freshness)
    elif observed_at is not None and quote_time is None:
        freshness = "unknown"
        if not any(item.startswith("quote_timestamp_") for item in blockers):
            blockers.append("freshness_unknown")

    levels_out = []
    touches = []
    for row in parsed:
        minimum = row["minimum"] if row["minimum"] is not None else Decimal("Infinity")
        touch_bid = _Touch(row["bids"], True, minimum) if row["minimum"] is not None else None
        touch_ask = _Touch(row["asks"], False, minimum) if row["minimum"] is not None else None
        if touch_bid and touch_ask:
            touches.append({"outcome": row["outcome"], "abbr": row["abbr"], "side": row["side"],
                            "bid": touch_bid["price_text"], "bid_size": touch_bid["size_text"],
                            "ask": touch_ask["price_text"], "ask_size": touch_ask["size_text"]})
        ordered_bids = sorted(row["bids"], key=lambda item: item["price"], reverse=True)[:5]
        ordered_asks = sorted(row["asks"], key=lambda item: item["price"])[:5]
        for level in ordered_bids:
            levels_out.append(_PublicLevel(level, row["outcome"], "bid", touch_bid is not None and level["price"] == touch_bid["price"]))
        for level in ordered_asks:
            levels_out.append(_PublicLevel(level, row["outcome"], "ask", touch_ask is not None and level["price"] == touch_ask["price"]))

    # Retrieval time is recorded separately and is never copied into the source timestamp.
    executable = not blockers and freshness == "fresh" and len(touches) == 2
    return {
        "venue": "polymarket",
        "price_role": "market_price_not_a_model_forecast",
        "executable": executable,
        "indicative_only": not executable,
        "freshness": freshness,
        "blockers": list(dict.fromkeys(blockers)),
        "quote_timestamp_utc": FormatUtc(quote_time) if quote_time else None,
        "quote_timestamp_et": FormatEt(quote_time) if quote_time else None,
        "quote_age_seconds": AgeSeconds(quote_time, observed_at) if observed_at and quote_time else None,
        "observed_at_utc": FormatUtc(observed_at) if observed_at else None,
        "observed_at_et": FormatEt(observed_at) if observed_at else None,
        "source_timestamps": [
            {"token_id": row["token_id"], "raw": row["raw_timestamp"], "kind": row["timestamp_kind"],
             "utc": FormatUtc(row["source_time"]) if row["source_time"] else None}
            for row in parsed
        ],
        "levels": levels_out,
        "touches": touches,
        "books": [
            {"outcome": row["outcome"], "token_id": row["token_id"], "book_url": row["book_url"], "hash": row["hash"],
             "tick_size": row["tick_size"], "min_order_size": None if row["minimum"] is None else str(row["minimum"])}
            for row in parsed
        ],
    }


def GuardModels(pregame_model) -> dict:
    """Keep a supplied pregame probability visible and unusable for edge."""
    supplied = pregame_model is not None
    probability = None
    if isinstance(pregame_model, dict):
        probability = pregame_model.get("home_prob")
    return {
        "pregame_model": {
            "available": supplied,
            "scope": "pregame_only",
            "usable_as_inplay": False,
            "name": (pregame_model or {}).get("name", "elo_pregame") if supplied else None,
            "home_prob": probability,
            "note": (
                "Supplied pregame probability is labeled and ignored for edge. It is not an in-play forecast."
                if supplied else
                "Backtest.PredictDay Elo is pregame only and is not consulted by this report."),
        },
        "inplay_model": {
            "available": False,
            "validated": False,
            "historical_state_odds_validation": False,
            "calibrated_live_probability": None,
            "probability": None,
            "reason": (
                "No validated in-play win model exists. Pregame Elo backtests are not a live "
                "calibration, and a live probability is not invented from them."),
        },
        "market_implied_probability_is_model_forecast": False,
        "supported_edge": None,
        "expected_value": None,
        "assessment": "unavailable",
        "assessment_reason": (
            "Unvalidated live model: no supported edge. A market implied probability is not a model forecast. "
            "Pregame Elo is not a substitute."),
    }


def Readiness(book: dict, game_state: dict) -> str:
    if book.get("executable") and game_state.get("usable"):
        return "executable_quote"
    if book.get("executable"):
        return "executable_unaligned"
    if book.get("blockers"):
        return "not_executable"
    return "blocked"


def PhaseFromScoreboard(game: dict) -> str:
    text = _Text(game.get("gameStatusText")).strip().lower()
    try:
        period = int(game.get("period") or 0)
    except (TypeError, ValueError):
        period = 0
    try:
        regulation = int(game.get("regulationPeriods") or 4)
    except (TypeError, ValueError):
        regulation = 4
    status = game.get("gameStatus")
    if status == 3 or text.startswith("final"):
        return "final"
    if period > regulation or text.startswith("ot") or " ot" in f" {text}":
        return "overtime"
    if status == 2 or period >= 1:
        return "in_progress"
    if status == 1 or period == 0:
        return "scheduled"
    return "unknown"


def EvaluateGameState(game, resolved_time: dict, observed_at, scoreboard_date: str, requested_date: str) -> dict:
    """NBA scoreboard snapshot. usable only when the slate date matches and the snapshot time is fresh."""
    blockers = []
    if game is None:
        return {"source": "nba_cdn_scoreboard", "usable": False, "freshness": "unknown",
                "phase": "unknown", "blockers": ["game_missing_on_scoreboard"],
                "observed_at_utc": FormatUtc(observed_at) if observed_at else None}
    if scoreboard_date != requested_date:
        blockers.append("scoreboard_date_mismatch")
    phase = PhaseFromScoreboard(game)
    if phase == "unknown":
        blockers.append("game_phase_unknown")
    freshness = resolved_time.get("freshness") or "unknown"
    if freshness != "fresh":
        blockers.append("game_state_" + freshness)
    home = (game.get("homeTeam") or {})
    away = (game.get("awayTeam") or {})
    try:
        home_score = int(home.get("score"))
        away_score = int(away.get("score"))
    except (TypeError, ValueError):
        home_score = away_score = None
        blockers.append("score_missing")
    if phase in ("in_progress", "overtime", "final") and (home_score is None or away_score is None):
        blockers.append("score_missing")
    usable = not blockers
    moment = resolved_time.get("utc")
    return {
        "source": "nba_cdn_scoreboard",
        "source_url": "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
        "documented_via": "nba_api.live.nba.endpoints.scoreboard.ScoreBoard",
        "usable": usable,
        "freshness": freshness,
        "timezone_resolution": resolved_time.get("timezone_resolution"),
        "raw_meta_time": resolved_time.get("raw"),
        "timestamp_utc": FormatUtc(moment) if moment else None,
        "timestamp_et": resolved_time.get("et"),
        "age_seconds": AgeSeconds(moment, observed_at) if observed_at and moment else None,
        "observed_at_utc": FormatUtc(observed_at) if observed_at else None,
        "observed_at_et": FormatEt(observed_at) if observed_at else None,
        "phase": phase,
        "overtime": phase == "overtime",
        "regulation_periods": game.get("regulationPeriods"),
        "period": game.get("period"),
        "clock": game.get("gameClock"),
        "status_text": game.get("gameStatusText"),
        "game_id": game.get("gameId"),
        "home": {"abbr": (home.get("teamTricode") or "").upper(), "name": home.get("teamName"), "score": home_score},
        "away": {"abbr": (away.get("teamTricode") or "").upper(), "name": away.get("teamName"), "score": away_score},
        "blockers": blockers,
        "note": "meta.time has no offset. A naive clock stays freshness unknown. It is not assigned UTC or Eastern by comparing it with the observation.",
    }


def LatestOffsetPlay(body: dict):
    """Latest play whose timeActual has an explicit offset. Naive clocks are skipped."""
    actions = ((body or {}).get("game") or {}).get("actions") or []
    chosen = None
    for action in actions:
        if not isinstance(action, dict):
            continue
        moment, _kind = ParseInstant(action.get("timeActual"))
        if moment is None:
            continue
        try:
            number = int(action.get("actionNumber"))
        except (TypeError, ValueError):
            number = -1
        if chosen is None or (moment, number) > chosen[0]:
            chosen = ((moment, number), action, moment)
    if chosen is None:
        return None
    return {"action": chosen[1], "utc": chosen[2]}


def GameStateFromPlay(play: dict, observed_at, scoreboard_game: dict, scoreboard_date: str, requested_date: str) -> dict:
    """Score and clock from one play-by-play action. timeActual is the as-of, not meta.time."""
    blockers = []
    action, moment = play["action"], play["utc"]
    if scoreboard_date != requested_date:
        blockers.append("scoreboard_date_mismatch")
    freshness = Freshness(moment, observed_at, MAX_GAME_STATE_AGE_SECONDS) if observed_at else "unknown"
    if freshness != "fresh":
        blockers.append("game_state_" + freshness)
    phase = PhaseFromScoreboard(scoreboard_game) if scoreboard_game else "unknown"
    period_type = _Text(action.get("periodType")).upper()
    try:
        period = int(action.get("period") or 0)
    except (TypeError, ValueError):
        period = 0
        blockers.append("period_missing")
    try:
        regulation = int((scoreboard_game or {}).get("regulationPeriods") or 4)
    except (TypeError, ValueError):
        regulation = 4
    if phase != "final" and (period_type == "OVERTIME" or period > regulation):
        phase = "overtime"
    try:
        home_score = int(str(action.get("scoreHome")))
        away_score = int(str(action.get("scoreAway")))
    except (TypeError, ValueError):
        home_score = away_score = None
        blockers.append("score_missing")
    home = (scoreboard_game or {}).get("homeTeam") or {}
    away = (scoreboard_game or {}).get("awayTeam") or {}
    game_id = (scoreboard_game or {}).get("gameId")
    return {
        "source": "nba_cdn_playbyplay",
        "source_url": f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json",
        "documented_via": "nba_api.live.nba.endpoints.playbyplay.PlayByPlay timeActual",
        "usable": not blockers,
        "freshness": freshness,
        "timezone_resolution": "explicit_offset",
        "raw_time_actual": action.get("timeActual"),
        "timestamp_utc": FormatUtc(moment),
        "timestamp_et": FormatEt(moment),
        "age_seconds": AgeSeconds(moment, observed_at) if observed_at else None,
        "observed_at_utc": FormatUtc(observed_at) if observed_at else None,
        "observed_at_et": FormatEt(observed_at) if observed_at else None,
        "phase": phase,
        "overtime": phase == "overtime",
        "regulation_periods": (scoreboard_game or {}).get("regulationPeriods"),
        "period": action.get("period"),
        "period_type": action.get("periodType"),
        "clock": action.get("clock"),
        "status_text": (scoreboard_game or {}).get("gameStatusText"),
        "action_number": action.get("actionNumber"),
        "game_id": game_id,
        "home": {"abbr": (home.get("teamTricode") or "").upper(), "name": home.get("teamName"), "score": home_score},
        "away": {"abbr": (away.get("teamTricode") or "").upper(), "name": away.get("teamName"), "score": away_score},
        "blockers": blockers,
        "scoreboard_meta_time_not_used": True,
        "note": "timeActual has an explicit offset and is the score as-of. Scoreboard meta.time is not used.",
    }


def SynchronizeQuoteAndState(state: dict, quote_time_utc) -> dict:
    """Refuse alignment when the quote snapshot and the score as-of are too far apart."""
    if not state:
        return state
    state = dict(state)
    blockers = list(state.get("blockers") or [])
    quote_time, _kind = ParseInstant(quote_time_utc) if quote_time_utc else (None, "missing")
    state_time, _kind = ParseInstant(state.get("timestamp_utc"))
    if quote_time is None or state_time is None:
        if state.get("usable"):
            state["usable"] = False
            blockers.append("quote_state_not_synchronized")
        state["blockers"] = blockers
        state["quote_state_gap_seconds"] = None
        return state
    gap = abs((quote_time - state_time).total_seconds())
    state["quote_state_gap_seconds"] = round(gap, 3)
    if gap > MAX_QUOTE_STATE_SKEW_SECONDS:
        state["usable"] = False
        blockers.append("quote_state_not_synchronized")
    state["blockers"] = blockers
    return state


def RejectUnverifiedAggregate(quote: dict) -> dict:
    """ESPN-style consensus numbers are not an executable venue book."""
    has_contract = bool(quote.get("condition_id") or quote.get("ticker") or quote.get("venue_url"))
    has_book = bool(quote.get("bids") or quote.get("asks") or quote.get("levels"))
    return {
        "executable": False,
        "indicative_only": True,
        "tag": "unverified_aggregate",
        "reason": "No venue contract id and no verified bid/ask depth.",
        "has_contract": has_contract,
        "has_book": has_book,
        "price_role": "market_price_not_a_model_forecast",
    }
