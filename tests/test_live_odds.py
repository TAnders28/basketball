"""Labeled examples only. None of these payloads is a live quote.

Run `python -m LiveOdds.Smoke` to observe the public sources. A fixture cannot
establish live readiness.
"""

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from LiveOdds.Report import BuildReport
from LiveOdds.Sources import ClassifyPolymarketMarket, LoadBundle, ReadOnlyHttp
from LiveOdds.Teams import ResolveTeam
from LiveOdds.Time import FormatEt, ParseInstant, ResolveNaiveWallClock
from LiveOdds.Validate import (
    EvaluatePolymarketBook, FeeStatus, GuardModels, RejectUnverifiedAggregate,
)

EXAMPLE = "EXAMPLE_NOT_A_LIVE_QUOTE"
OBSERVED = datetime(2026, 11, 9, 1, 42, 35, tzinfo=timezone.utc)
FRESH = OBSERVED - timedelta(seconds=1)
ROOT = Path(__file__).resolve().parents[1]


def _ms(moment):
    return str(int(moment.timestamp() * 1000))


def _raw(**overrides):
    market = {
        "_label": EXAMPLE,
        "sportsMarketType": "moneyline",
        "slug": "nba-uta-min-2026-11-08",
        "question": "Jazz vs. Timberwolves",
        "conditionId": "0xabc",
        "outcomes": json.dumps(["Jazz", "Timberwolves"]),
        "clobTokenIds": json.dumps(["111", "222"]),
        "gameStartTime": "2026-11-08 23:00:00+00",
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "archived": False,
        "umaResolutionStatus": None,
        "description": (
            "If the Jazz win, the market will resolve to \"Jazz\".\n"
            "The result will be determined based on the final score including any overtime periods."
        ),
        "resolutionSource": "https://www.nba.com/",
        "feesEnabled": True,
        "feeType": "sports_fees_v3",
        "feeSchedule": {"exponent": 1, "rate": 0.05, "takerOnly": True, "rebateRate": 0.15},
        "makerBaseFee": 1000,
        "takerBaseFee": 1000,
        "bestBid": 0.40,
        "bestAsk": 0.42,
        "outcomePrices": json.dumps(["0.41", "0.59"]),
        "events": [{
            "eventDate": "2026-11-08",
            "score": "63-68",
            "period": "Q3",
            "elapsed": "06:06",
            "live": True,
            "ended": False,
            "updatedAt": "2026-11-09T01:37:50Z",
            "gameId": "example-game",
        }],
    }
    market.update(overrides)
    return market


def _book(token, bids, asks, moment=FRESH, condition="0xabc", minimum="5"):
    return {
        "body": {
            "market": condition,
            "asset_id": token,
            "timestamp": _ms(moment),
            "hash": EXAMPLE,
            "bids": [{"price": price, "size": size} for price, size in bids],
            "asks": [{"price": price, "size": size} for price, size in asks],
            "min_order_size": minimum,
            "tick_size": "0.01",
        },
        "observed_at": OBSERVED,
        "url": f"https://clob.polymarket.com/book?token_id={token}",
    }


def _books(moment=FRESH):
    # Worst price first, so a correct reader has to sort.
    return {
        "111": _book("111", [("0.10", "20"), ("0.40", "100")], [("0.90", "15"), ("0.42", "80")], moment),
        "222": _book("222", [("0.10", "15"), ("0.58", "80")], [("0.99", "20"), ("0.60", "100")], moment),
    }


def _fees():
    return {
        "111": {"body": {"base_fee": 1000}, "observed_at": OBSERVED, "url": "https://clob.polymarket.com/fee-rate?token_id=111"},
        "222": {"body": {"base_fee": 1000}, "observed_at": OBSERVED, "url": "https://clob.polymarket.com/fee-rate?token_id=222"},
    }


def _scoreboard(meta_time, status=2, text="3rd Qtr", period=3, away="UTA", home="MIN"):
    return {
        "error": None,
        "observed_at": OBSERVED,
        "url": "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
        "body": {
            "meta": {"time": meta_time, "code": 200},
            "scoreboard": {
                "gameDate": "2026-11-08",
                "games": [{
                    "gameId": "0022600999",
                    "gameStatus": status,
                    "gameStatusText": text,
                    "period": period,
                    "regulationPeriods": 4,
                    "gameClock": "PT05M33.00S",
                    "awayTeam": {"teamTricode": away, "teamName": "Jazz", "score": 63},
                    "homeTeam": {"teamTricode": home, "teamName": "Timberwolves", "score": 70},
                }],
            },
        },
    }


def _bundle(market=None, markets=None, meta="2026-11-08 20:42:32.4232", books=None, pregame=None,
            scoreboard=None, polymarket_error=None, kalshi_markets=None, kalshi_books=None):
    return {
        "date": "2026-11-08",
        "live_sample": False,
        "pregame_model": pregame,
        "polymarket_error": polymarket_error,
        "polymarket_markets": [] if polymarket_error else (markets if markets is not None else [market or _raw()]),
        "books": _books() if books is None else books,
        "fees": _fees(),
        "scoreboard_error": None if scoreboard is None or scoreboard.get("error") is None else scoreboard.get("error"),
        "scoreboard": scoreboard if scoreboard is not None else _scoreboard(meta),
        "kalshi_error": None,
        "kalshi_markets": kalshi_markets or [],
        "kalshi_books": kalshi_books or {},
    }


def _assert_no_edge(report):
    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("supported_edge", "expected_value", "fee_adjusted_probability"):
                    if value is not None:
                        raise AssertionError(key)
                if key == "usable_as_inplay" and value is not False:
                    raise AssertionError(key)
                if key == "market_implied_probability_is_model_forecast" and value is not False:
                    raise AssertionError(key)
                if key == "assumed_zero" and value is not False:
                    raise AssertionError(key)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(report)
    if report["assessment_tier"] != "unavailable":
        raise AssertionError(report["assessment_tier"])


class LiveOddsTests(unittest.TestCase):
    def test_example_book_is_executable_but_has_no_edge_and_naive_clock_stays_unknown(self):
        report = BuildReport(_bundle(pregame={"name": "elo_pregame", "home_prob": 0.81}))
        json.dumps(report)
        self.assertTrue(report["fixture"])
        self.assertFalse(report["live_sample"])
        _assert_no_edge(report)
        game = report["games"][0]
        self.assertEqual(game["readiness_tier"], "executable_unaligned")
        self.assertEqual(game["assessment_tier"], "unavailable")
        self.assertTrue(game["quote"]["executable"])
        self.assertEqual(game["quote"]["freshness"], "fresh")
        self.assertEqual(game["quote"]["price_role"], "market_price_not_a_model_forecast")
        self.assertNotEqual(game["quote"]["quote_timestamp_utc"], game["quote"]["observed_at_utc"])
        touches = {row["abbr"]: row for row in game["quote"]["touches"]}
        self.assertEqual(touches["UTA"]["bid"], "0.40")
        self.assertEqual(touches["UTA"]["ask"], "0.42")
        self.assertEqual(touches["MIN"]["bid"], "0.58")
        self.assertEqual(touches["MIN"]["ask"], "0.60")
        self.assertEqual(game["identity"]["event"], "https://polymarket.com/event/nba-uta-min-2026-11-08")
        self.assertEqual(game["identity"]["condition_id"], "0xabc")
        self.assertEqual(game["indicative_gamma"]["tag"], "indicative")
        self.assertEqual(game["fees"]["status"], "unreconciled")
        self.assertIsNone(game["fees"]["single_fee_rate"])
        self.assertFalse(game["fees"]["reconciled"])
        self.assertTrue(game["fees"]["maker_fields_unreconciled"])
        self.assertIsNone(game["fees"]["fee_adjusted_probability"])
        self.assertIsNone(game["models"]["inplay_model"]["calibrated_live_probability"])
        self.assertFalse(game["models"]["inplay_model"]["historical_state_odds_validation"])
        self.assertTrue(game["settlement"]["includes_overtime"])
        self.assertEqual(game["game_state"]["freshness"], "unknown")
        self.assertEqual(game["game_state"]["timezone_resolution"], "naive_no_offset")
        self.assertIsNone(game["game_state"]["timestamp_utc"])
        self.assertFalse(game["game_state"]["usable"])
        self.assertFalse(game["models"]["pregame_model"]["usable_as_inplay"])
        self.assertEqual(game["models"]["pregame_model"]["home_prob"], 0.81)
        self.assertFalse(game["models"]["inplay_model"]["available"])
        self.assertEqual(game["venue_reported_state"]["score_team_attribution"], "unknown")

    def test_naive_clock_near_eastern_or_plus_12_is_not_accepted(self):
        for raw in ("2026-11-08 20:42:32.4232", "2026-11-08 08:42:32.4232"):
            resolved = ResolveNaiveWallClock(raw, OBSERVED)
            self.assertIsNone(resolved["utc"])
            self.assertEqual(resolved["freshness"], "unknown")
            self.assertEqual(resolved["timezone_resolution"], "naive_no_offset")
            self.assertIn("not_used", json.dumps(resolved))

    def test_explicit_offset_can_align_without_creating_an_edge(self):
        report = BuildReport(_bundle(meta="2026-11-09T01:42:30+00:00"))
        game = report["games"][0]
        self.assertEqual(game["game_state"]["timezone_resolution"], "explicit_offset")
        self.assertEqual(game["game_state"]["freshness"], "fresh")
        self.assertTrue(game["game_state"]["usable"])
        self.assertEqual(game["readiness_tier"], "executable_quote")
        self.assertEqual(game["assessment_tier"], "unavailable")
        self.assertIsNone(report["supported_edge"])
        self.assertIsNone(report["expected_value"])

    def test_stale_missing_and_future_timestamps_fail_closed(self):
        stale = BuildReport(_bundle(meta="2026-11-09T01:42:30+00:00", books=_books(OBSERVED - timedelta(seconds=120))))
        self.assertIn("stale", stale["games"][0]["quote"]["blockers"])
        self.assertFalse(stale["games"][0]["quote"]["executable"])
        self.assertEqual(stale["games"][0]["readiness_tier"], "not_executable")

        missing = _books()
        for token in missing:
            missing[token]["body"]["timestamp"] = None
        missing_report = BuildReport(_bundle(books=missing))
        quote = missing_report["games"][0]["quote"]
        self.assertIsNone(quote["quote_timestamp_utc"])
        self.assertIsNotNone(quote["observed_at_utc"])
        self.assertNotEqual(quote["quote_timestamp_utc"], quote["observed_at_utc"])
        self.assertEqual(quote["freshness"], "unknown")
        self.assertFalse(quote["executable"])

        skewed = _books(OBSERVED + timedelta(seconds=60))
        skewed_report = BuildReport(_bundle(meta="2026-11-09T01:42:30+00:00", books=skewed))
        self.assertIn("clock_skew", skewed_report["games"][0]["quote"]["blockers"])
        self.assertFalse(skewed_report["games"][0]["quote"]["executable"])

    def test_pregame_window_is_looser_only_when_state_is_verified_scheduled(self):
        books = _books(OBSERVED - timedelta(seconds=60))
        scheduled = _scoreboard("2026-11-09T01:42:30+00:00", status=1, text="7:00 pm ET", period=0)
        fresh = BuildReport(_bundle(books=books, scoreboard=scheduled))
        self.assertEqual(fresh["games"][0]["quote"]["freshness"], "fresh")
        self.assertTrue(fresh["games"][0]["quote"]["executable"])
        live = BuildReport(_bundle(meta="2026-11-09T01:42:30+00:00", books=books))
        self.assertIn("stale", live["games"][0]["quote"]["blockers"])

    def test_timezone_presentation_uses_eastern(self):
        moment, kind = ParseInstant("2026-11-09T01:00:00Z")
        self.assertEqual(kind, "iso")
        self.assertEqual(FormatEt(moment), "2026-11-08 20:00:00 ET")
        tip, tip_kind = ParseInstant("2026-11-08 23:00:00+00")
        self.assertEqual(tip_kind, "iso")
        self.assertEqual(FormatEt(tip), "2026-11-08 18:00:00 ET")

    def test_book_timestamp_gap_and_crossed_or_empty_books(self):
        gap = _books()
        gap["222"]["body"]["timestamp"] = _ms(FRESH + timedelta(seconds=5))
        gapped = BuildReport(_bundle(meta="2026-11-09T01:42:30+00:00", books=gap))
        self.assertIn("book_timestamp_mismatch", gapped["games"][0]["quote"]["blockers"])

        crossed = _books()
        crossed["111"]["body"]["bids"] = [{"price": "0.50", "size": "100"}]
        crossed["111"]["body"]["asks"] = [{"price": "0.42", "size": "80"}]
        self.assertIn("crossed_book", BuildReport(_bundle(books=crossed))["games"][0]["quote"]["blockers"])

        complement = _books()
        complement["111"]["body"]["bids"] = [{"price": "0.70", "size": "100"}]
        complement["111"]["body"]["asks"] = [{"price": "0.72", "size": "80"}]
        complement["222"]["body"]["bids"] = [{"price": "0.40", "size": "80"}]
        complement["222"]["body"]["asks"] = [{"price": "0.42", "size": "100"}]
        self.assertIn("crossed_complement", BuildReport(_bundle(books=complement))["games"][0]["quote"]["blockers"])

        empty = _books()
        empty["111"]["body"]["asks"] = []
        self.assertIn("missing_liquidity", BuildReport(_bundle(books=empty))["games"][0]["quote"]["blockers"])

        dust = _books()
        for token in dust:
            dust[token]["body"]["bids"] = [{"price": "0.40", "size": "1"}]
            dust[token]["body"]["asks"] = [{"price": "0.42", "size": "1"}]
        self.assertIn("depth_below_min_order", BuildReport(_bundle(books=dust))["games"][0]["quote"]["blockers"])

    def test_mapping_failures(self):
        swapped = _scoreboard("2026-11-09T01:42:30+00:00", away="MIN", home="UTA")
        swap = BuildReport(_bundle(scoreboard=swapped))
        self.assertIn("home_away_mismatch", swap["games"][0]["blockers"])
        self.assertNotEqual(swap["games"][0]["readiness_tier"], "executable_quote")

        other = _raw(slug="nba-bos-mil-2026-11-08", outcomes=json.dumps(["Celtics", "Bucks"]),
                     question="Celtics vs. Bucks")
        missing_team = BuildReport(_bundle(markets=[other]))
        self.assertIn("team_not_on_scoreboard", missing_team["games"][0]["blockers"])

        mismatch = _raw(gameStartTime="2026-11-10T05:00:00Z")
        self.assertEqual(ClassifyPolymarketMarket(mismatch, "2026-11-08")["reason"], "date_mismatch")
        dated = BuildReport(_bundle(markets=[mismatch]))
        self.assertIn("date_mismatch", dated["games"][0]["blockers"])
        self.assertEqual(dated["games"][0]["readiness_tier"], "blocked")

        bad_names = _raw(outcomes=json.dumps(["Celtics", "Bucks"]))
        self.assertEqual(ClassifyPolymarketMarket(bad_names, "2026-11-08")["reason"], "outcome_team_mismatch")

        duplicate = [_raw(conditionId="0xabc"), _raw(conditionId="0xdef", question="Jazz vs. Timberwolves duplicate")]
        ambiguous = BuildReport(_bundle(markets=duplicate))
        self.assertTrue(any("ambiguous_event_mapping" in game["blockers"] for game in ambiguous["games"]))
        self.assertTrue(all(not (game.get("quote") or {}).get("executable") for game in ambiguous["games"]))

        self.assertEqual(ResolveTeam("Los Angeles")["reason"], "ambiguous_team")
        self.assertEqual(ResolveTeam("Los Angeles L")["abbr"], "LAL")
        self.assertEqual(ResolveTeam("Los Angeles C")["abbr"], "LAC")
        self.assertEqual(ResolveTeam("Globetrotters")["reason"], "unknown_team")

    def test_closed_market_and_overtime_settlement(self):
        closed = BuildReport(_bundle(market=_raw(closed=True, acceptingOrders=False)))
        self.assertIn("suspended_or_closed", closed["games"][0]["quote"]["blockers"])
        proposed = BuildReport(_bundle(market=_raw(umaResolutionStatus="proposed")))
        self.assertTrue(any(item.startswith("resolution_") for item in proposed["games"][0]["quote"]["blockers"]))

        overtime = _scoreboard("2026-11-09T01:42:30+00:00", status=2, text="OT 1", period=5)
        included = BuildReport(_bundle(scoreboard=overtime))
        self.assertTrue(included["games"][0]["quote"]["executable"])
        self.assertTrue(included["games"][0]["game_state"]["overtime"])
        self.assertEqual(included["games"][0]["assessment_tier"], "unavailable")

        excluded = _raw(description="Regulation only. This market excludes overtime.")
        excluded_report = BuildReport(_bundle(market=excluded, scoreboard=overtime))
        self.assertIn("overtime_settlement_excludes_ot", excluded_report["games"][0]["quote"]["blockers"])

        silent = _raw(description="If the Jazz win, the market resolves to Jazz.")
        silent_report = BuildReport(_bundle(market=silent, scoreboard=overtime))
        self.assertIn("overtime_settlement_unknown", silent_report["games"][0]["quote"]["blockers"])
        self.assertIsNone(silent_report["games"][0]["settlement"]["includes_overtime"])
        redemption = silent_report["games"][0]["settlement"]["platform_redemption"]
        self.assertEqual(redemption["winning_token"], "1 pUSD")
        self.assertFalse(redemption["decides_overtime"])

    def test_fees_are_not_converted_or_assumed_zero(self):
        unknown = FeeStatus({}, {}, ["111"])
        self.assertEqual(unknown["status"], "unknown")
        self.assertIsNone(unknown["single_fee_rate"])
        self.assertFalse(unknown["assumed_zero"])
        self.assertIsNone(unknown["fee_adjusted_probability"])

        # 0.03 and 300 bps would match only under an undocumented conversion. Do not apply it.
        both = FeeStatus({
            "fee_schedule": {"rate": "0.03", "taker_only": True},
            "fees_enabled": True, "maker_base_fee": 0, "taker_base_fee": 300,
        }, {"111": {"base_fee": 300}}, ["111"])
        self.assertEqual(both["status"], "unreconciled")
        self.assertIsNone(both["single_fee_rate"])
        self.assertFalse(both["reconciled"])
        self.assertIsNone(both["fee_adjusted_probability"])

        curve_and_bps = FeeStatus({
            "fee_schedule": {"rate": "0.05", "exponent": 1, "taker_only": True},
            "fees_enabled": True, "maker_base_fee": 1000, "taker_base_fee": 1000,
        }, {"111": {"base_fee": 1000}}, ["111"])
        self.assertEqual(curve_and_bps["status"], "unreconciled")
        self.assertIsNone(curve_and_bps["single_fee_rate"])
        self.assertTrue(curve_and_bps["maker_fields_unreconciled"])

        zero = FeeStatus({
            "fee_schedule": {"rate": 0, "taker_only": True},
            "fees_enabled": False, "maker_base_fee": 0, "taker_base_fee": 0,
        }, {"111": {"base_fee": 0}}, ["111"])
        self.assertEqual(zero["status"], "explicit_zero")
        self.assertTrue(zero["source_says_zero"])
        self.assertFalse(zero["assumed_zero"])
        self.assertIsNone(zero["single_fee_rate"])

        not_zero = FeeStatus({
            "fee_schedule": {"rate": 0, "taker_only": True},
            "fees_enabled": False, "maker_base_fee": 0,
        }, {"111": {"base_fee": 1000}}, ["111"])
        self.assertEqual(not_zero["status"], "unreconciled")
        self.assertFalse(not_zero["source_says_zero"])
        self.assertFalse(not_zero["assumed_zero"])

    def test_offset_play_aligns_score_without_creating_an_edge(self):
        moment = OBSERVED - timedelta(seconds=2)
        stamped = moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        bundle = _bundle(meta="2026-11-08 08:42:32.4232")
        bundle["play_by_play"] = {"0022600999": {
            "observed_at": OBSERVED, "error": None,
            "body": {"game": {"actions": [
                {"actionNumber": 1, "timeActual": "2026-11-08 20:00:00", "scoreHome": "1", "scoreAway": "1",
                 "period": 1, "periodType": "REGULAR", "clock": "PT11M00.00S"},
                {"actionNumber": 2, "timeActual": stamped, "scoreHome": "70", "scoreAway": "63",
                 "period": 3, "periodType": "REGULAR", "clock": "PT05M33.00S"},
            ]}},
        }}
        game = BuildReport(bundle)["games"][0]
        self.assertEqual(game["readiness_tier"], "executable_quote")
        self.assertEqual(game["assessment_tier"], "unavailable")
        self.assertEqual(game["game_state"]["source"], "nba_cdn_playbyplay")
        self.assertEqual(game["game_state"]["timezone_resolution"], "explicit_offset")
        self.assertTrue(game["game_state"]["scoreboard_meta_time_not_used"])
        self.assertEqual(game["game_state"]["home"]["score"], 70)
        self.assertEqual(game["game_state"]["away"]["score"], 63)
        self.assertIsNone(game["models"]["inplay_model"]["calibrated_live_probability"])
        self.assertLessEqual(game["game_state"]["quote_state_gap_seconds"], 30)

        naive = dict(bundle)
        naive["play_by_play"] = {"0022600999": {
            "observed_at": OBSERVED, "error": None,
            "body": {"game": {"actions": [
                {"actionNumber": 2, "timeActual": "2026-11-08 20:57:20", "scoreHome": "70", "scoreAway": "63",
                 "period": 3, "periodType": "REGULAR"},
            ]}},
        }}
        naive_game = BuildReport(naive)["games"][0]
        self.assertEqual(naive_game["game_state"]["freshness"], "unknown")
        self.assertFalse(naive_game["game_state"]["usable"])
        self.assertNotEqual(naive_game["readiness_tier"], "executable_quote")
        self.assertTrue(naive_game["game_state"]["scoreboard_meta_time_not_used"])

        stale_at = (OBSERVED - timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        late = dict(bundle)
        late["play_by_play"] = {"0022600999": {
            "observed_at": OBSERVED, "error": None,
            "body": {"game": {"actions": [
                {"actionNumber": 9, "timeActual": stale_at, "scoreHome": "70", "scoreAway": "63",
                 "period": 3, "periodType": "REGULAR"},
            ]}},
        }}
        late_game = BuildReport(late)["games"][0]
        self.assertNotEqual(late_game["readiness_tier"], "executable_quote")
        self.assertTrue(set(late_game["game_state"]["blockers"]) & {"game_state_stale", "quote_state_not_synchronized"})

    def test_kalshi_metadata_time_is_not_a_quote_time(self):
        markets = []
        for side, title, bid in (("UTA", "Utah", "0.4000"), ("MIN", "Minnesota", "0.5800")):
            markets.append({
                "ticker": f"KXNBAGAME-26NOV08UTAMIN-{side}",
                "event_ticker": "KXNBAGAME-26NOV08UTAMIN",
                "status": "active",
                "yes_sub_title": title,
                "rules_primary": (
                    f"If {title} wins the Utah vs Minnesota Pro Basketball game originally "
                    "scheduled for Nov 8, 2026, then the market resolves to Yes."
                ),
                "rules_secondary": "Postponement language with no overtime clause.",
                "yes_bid_dollars": bid,
                "yes_ask_dollars": "0.4200" if side == "UTA" else "0.6000",
                "yes_bid_size_fp": "100.00",
                "notional_value_dollars": "1.0000",
                "updated_time": "2026-11-09T01:42:35Z",
            })
        books = {}
        for market in markets:
            books[market["ticker"]] = {
                "observed_at": OBSERVED,
                "url": market["ticker"],
                "body": {"orderbook_fp": {"yes_dollars": [[market["yes_bid_dollars"], "100.00"]], "no_dollars": [["0.1000", "10.00"]]}},
            }
        # updated_time is exactly the observation and must still not become the quote time.
        report = BuildReport(_bundle(kalshi_markets=markets, kalshi_books=books))
        kalshi = report["games"][0]["kalshi"]
        self.assertFalse(kalshi["executable"])
        self.assertEqual(kalshi["freshness"], "unknown")
        self.assertEqual(kalshi["fees"]["status"], "unknown")
        self.assertFalse(kalshi["fees"]["assumed_zero"])
        self.assertIsNone(kalshi["settlement"]["includes_overtime"])
        for contract in kalshi["contracts"]:
            self.assertIsNone(contract["quote_timestamp_utc"])
            self.assertFalse(contract["metadata_is_quote_timestamp"])
            self.assertTrue(contract["derived_no_asks"])
            self.assertEqual(contract["derived_no_asks"][0]["ask_source"], "kalshi_documented_complement")

    def test_unverified_aggregate_and_source_outage_invent_nothing(self):
        rejected = RejectUnverifiedAggregate({"source": "espn", "price": 0.62})
        self.assertFalse(rejected["executable"])
        self.assertEqual(rejected["tag"], "unverified_aggregate")
        outage = BuildReport(_bundle(polymarket_error="source_outage"))
        self.assertEqual(outage["readiness_tier"], "blocked")
        self.assertEqual(outage["games"], [])
        self.assertIn("polymarket_source_outage", outage["blockers"])
        self.assertIsNone(outage["supported_edge"])
        _assert_no_edge(outage)

    def test_guard_models_and_direct_book_do_not_treat_price_as_forecast(self):
        guarded = GuardModels({"name": "elo_pregame", "home_prob": 0.77})
        self.assertFalse(guarded["inplay_model"]["available"])
        self.assertIsNone(guarded["supported_edge"])
        self.assertFalse(guarded["market_implied_probability_is_model_forecast"])
        market = ClassifyPolymarketMarket(_raw(), "2026-11-08")["market"]
        market["includes_overtime"] = True
        evaluated = EvaluatePolymarketBook(market, _books(), "in_progress")
        self.assertTrue(evaluated["executable"])
        self.assertEqual(evaluated["price_role"], "market_price_not_a_model_forecast")

    def test_client_is_get_only_and_refuses_other_hosts(self):
        calls = []

        def get(url, headers, timeout):
            calls.append(url)
            return 200, {"ok": True}

        http = ReadOnlyHttp(get=get, clock=lambda: OBSERVED)
        blocked = http.GetJson("http://clob.polymarket.com/book")
        self.assertEqual(blocked["error"], "host_not_allowed")
        self.assertEqual(calls, [])
        blocked = http.GetJson("https://example.com/order")
        self.assertEqual(blocked["error"], "host_not_allowed")
        text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "LiveOdds").glob("*.py"))
        for banned in ("requests.post", "requests.put", "requests.delete", "place_order", "PlaceOrder"):
            self.assertNotIn(banned, text)

    def test_load_bundle_uses_scripted_gets_only(self):
        routes = []

        def get(url, headers, timeout):
            routes.append(url)
            if url.endswith("/sports"):
                return 200, [{"sport": "nba", "primaryTagId": 745}]
            if "markets/keyset" in url:
                return 200, {"markets": [_raw()], "next_cursor": None}
            if "/book?" in url:
                token = "111" if "111" in url else "222"
                return 200, _books()[token]["body"]
            if "/fee-rate?" in url:
                return 200, {"base_fee": 1000}
            if url.endswith("todaysScoreboard_00.json"):
                return 200, _scoreboard("2026-11-08 08:42:32.4232")["body"]
            if "kalshi.com" in url and "/markets?" in url:
                return 200, {"markets": [], "cursor": None}
            return 404, None

        bundle = LoadBundle("2026-11-08", http=ReadOnlyHttp(get=get, clock=lambda: OBSERVED), live_sample=False)
        report = BuildReport(bundle)
        self.assertTrue(routes)
        self.assertTrue(all(url.startswith("https://") for url in routes))
        self.assertNotEqual(report["games"][0]["readiness_tier"], "executable_quote")
        self.assertFalse(report["games"][0]["game_state"]["usable"])
        self.assertIsNone(report["games"][0]["game_state"].get("timestamp_utc"))
        self.assertEqual(report["assessment_tier"], "unavailable")
        self.assertFalse(report["live_sample"])


if __name__ == "__main__":
    unittest.main()
