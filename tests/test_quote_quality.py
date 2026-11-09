"""Offline quote-schema / freshness regression tests.

Deterministic fixtures only (fixed aware clock). No network.
Documents existing LiveOdds thresholds: quote 30s, state 45s, quote-state skew 30s.
Does not change production code or loosen limits.
"""

from __future__ import annotations

import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from Backtest.AssessLive import RowsFromReport
from LiveOdds.Report import BuildReport
from LiveOdds.Time import (
    EASTERN,
    Freshness,
    MAX_FUTURE_SKEW_SECONDS,
    MAX_GAME_STATE_AGE_SECONDS,
    MAX_LIVE_QUOTE_AGE_SECONDS,
    MAX_QUOTE_STATE_SKEW_SECONDS,
    FormatUtc,
    ParseInstant,
)
from LiveOdds.Validate import SynchronizeQuoteAndState


def _LoadLiveOddsFixtures():
    path = Path(__file__).resolve().parents[1] / "tests" / "test_live_odds.py"
    spec = importlib.util.spec_from_file_location("live_odds_fixtures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIX = _LoadLiveOddsFixtures()
OBSERVED = FIX.OBSERVED  # fixed aware clock from existing suite


class QuoteQualityRegressionTests(unittest.TestCase):
    """Existing schema/freshness behavior — thresholds must stay 30 / 45 / 30."""

    def test_thresholds_documented_unchanged(self):
        self.assertEqual(MAX_LIVE_QUOTE_AGE_SECONDS, 30)
        self.assertEqual(MAX_GAME_STATE_AGE_SECONDS, 45)
        self.assertEqual(MAX_QUOTE_STATE_SKEW_SECONDS, 30)
        self.assertEqual(MAX_FUTURE_SKEW_SECONDS, 5)

    def test_control_well_formed_executable_and_aligned(self):
        report = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00"))
        game = report["games"][0]
        self.assertEqual(game["readiness_tier"], "executable_quote")
        self.assertTrue(game["quote"]["executable"])
        self.assertEqual(game["quote"]["freshness"], "fresh")
        self.assertTrue(game["game_state"]["usable"])
        self.assertEqual(game["game_state"]["freshness"], "fresh")
        self.assertEqual(game["assessment_tier"], "unavailable")
        self.assertFalse(game["models"]["inplay_model"]["available"])

    def test_control_assesslive_still_pass_null_model_edge(self):
        report = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00"))
        rows = RowsFromReport(report, MAX_LIVE_QUOTE_AGE_SECONDS, MAX_GAME_STATE_AGE_SECONDS)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(any(r.get("quote_qualified") for r in rows))
        for row in rows:
            self.assertEqual(row["action"], "PASS")
            self.assertEqual(row["reason"], "no live model")
            self.assertIsNone(row["model_prob"])
            self.assertIsNone(row["edge"])

    def test_missing_quote_timestamp(self):
        books = FIX._books()
        for token, wrapped in list(books.items()):
            body = dict(wrapped["body"])
            body.pop("timestamp", None)
            books[token] = {**wrapped, "body": body}
        game = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00", books=books))["games"][0]
        self.assertFalse(game["quote"]["executable"])
        self.assertEqual(game["quote"]["freshness"], "unknown")
        self.assertIn("quote_timestamp_missing", game["quote"]["blockers"])
        self.assertEqual(game["readiness_tier"], "not_executable")

    def test_malformed_quote_timestamp(self):
        books = FIX._books()
        for token, wrapped in list(books.items()):
            body = dict(wrapped["body"])
            body["timestamp"] = "not-a-time"
            books[token] = {**wrapped, "body": body}
        game = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00", books=books))["games"][0]
        self.assertFalse(game["quote"]["executable"])
        self.assertEqual(game["quote"]["freshness"], "unknown")
        self.assertIn("quote_timestamp_invalid", game["quote"]["blockers"])
        self.assertEqual(game["readiness_tier"], "not_executable")

    def test_naive_quote_instant_parse_is_not_assigned_a_zone(self):
        """Gap note: CLOB fixtures use unix ms; ParseInstant rejects naive ISO (kind=naive)."""
        moment, kind = ParseInstant("2026-11-09T01:42:30")
        self.assertIsNone(moment)
        self.assertEqual(kind, "naive")
        # Freshness with None source time stays unknown — must not substitute fetch/observed clock.
        self.assertEqual(Freshness(None, OBSERVED, MAX_LIVE_QUOTE_AGE_SECONDS), "unknown")

    def test_future_quote_timestamp_clock_skew(self):
        books = FIX._books(OBSERVED + timedelta(seconds=60))
        game = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00", books=books))["games"][0]
        self.assertFalse(game["quote"]["executable"])
        self.assertEqual(game["quote"]["freshness"], "clock_skew")
        self.assertIn("clock_skew", game["quote"]["blockers"])
        self.assertEqual(game["readiness_tier"], "not_executable")

    def test_stale_quote_over_30s(self):
        books = FIX._books(OBSERVED - timedelta(seconds=31))
        game = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00", books=books))["games"][0]
        self.assertFalse(game["quote"]["executable"])
        self.assertEqual(game["quote"]["freshness"], "stale")
        self.assertIn("stale", game["quote"]["blockers"])
        self.assertEqual(game["readiness_tier"], "not_executable")

    def test_missing_or_naive_state_timestamp(self):
        # Default fixture meta is naive wall clock → state freshness unknown / not usable.
        game = BuildReport(FIX._bundle())["games"][0]
        self.assertEqual(game["game_state"]["freshness"], "unknown")
        self.assertFalse(game["game_state"]["usable"])
        self.assertIsNone(game["game_state"].get("timestamp_utc"))
        self.assertEqual(game["readiness_tier"], "executable_unaligned")

    def test_stale_state_over_45s(self):
        # Observed ~01:42:35Z; meta 70s earlier → stale (>45).
        game = BuildReport(FIX._bundle(meta="2026-11-09T01:41:25+00:00"))["games"][0]
        self.assertEqual(game["game_state"]["freshness"], "stale")
        self.assertFalse(game["game_state"]["usable"])
        self.assertNotEqual(game["readiness_tier"], "executable_quote")

    def test_future_state_timestamp_clock_skew(self):
        game = BuildReport(FIX._bundle(meta="2026-11-09T01:43:30+00:00"))["games"][0]
        # 55s ahead of OBSERVED (01:42:35) exceeds MAX_FUTURE_SKEW_SECONDS (5).
        self.assertEqual(game["game_state"]["freshness"], "clock_skew")
        self.assertFalse(game["game_state"]["usable"])

    def test_alignment_skew_over_30s(self):
        report = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00"))
        state = dict(report["games"][0]["game_state"])
        quote_time = report["games"][0]["quote"]["quote_timestamp_utc"]
        # Place state >30s from the *source* quote clock (fixture books are OBSERVED-1s).
        state["timestamp_utc"] = FormatUtc(OBSERVED - timedelta(seconds=40))
        state["usable"] = True
        state["blockers"] = []
        synced = SynchronizeQuoteAndState(state, quote_time)
        self.assertFalse(synced["usable"])
        self.assertGreater(synced.get("quote_state_gap_seconds"), MAX_QUOTE_STATE_SKEW_SECONDS)
        self.assertIn("quote_state_not_synchronized", synced["blockers"])

    def test_alignment_within_30s_ok(self):
        report = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00"))
        state = dict(report["games"][0]["game_state"])
        quote_time = report["games"][0]["quote"]["quote_timestamp_utc"]
        state["timestamp_utc"] = FormatUtc(OBSERVED - timedelta(seconds=10))
        state["usable"] = True
        state["blockers"] = []
        synced = SynchronizeQuoteAndState(state, quote_time)
        self.assertTrue(synced["usable"])
        self.assertLessEqual(synced.get("quote_state_gap_seconds"), MAX_QUOTE_STATE_SKEW_SECONDS)

    def test_zero_and_negative_size_not_executable(self):
        zero_books = {
            "111": FIX._book("111", [("0.40", "100")], [("0.42", "0")]),
            "222": FIX._book("222", [("0.58", "80")], [("0.60", "100")]),
        }
        zero = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00", books=zero_books))["games"][0]
        self.assertFalse(zero["quote"]["executable"])
        self.assertEqual(zero["readiness_tier"], "not_executable")
        self.assertTrue(set(zero["quote"]["blockers"]) & {
            "depth_below_min_order", "missing_liquidity", "level_out_of_range",
        })

        neg_books = {
            "111": FIX._book("111", [("0.40", "100")], [("0.42", "-5")]),
            "222": FIX._book("222", [("0.58", "80")], [("0.60", "100")]),
        }
        neg = BuildReport(FIX._bundle(meta="2026-11-09T01:42:30+00:00", books=neg_books))["games"][0]
        self.assertFalse(neg["quote"]["executable"])
        self.assertEqual(neg["readiness_tier"], "not_executable")

    def test_equivalent_utc_and_et_instants_same_freshness(self):
        utc = datetime(2026, 11, 9, 23, 0, 0, tzinfo=timezone.utc)
        et = utc.astimezone(EASTERN)
        observed = utc + timedelta(seconds=1)
        self.assertEqual(
            Freshness(utc, observed, MAX_LIVE_QUOTE_AGE_SECONDS),
            Freshness(et, observed, MAX_LIVE_QUOTE_AGE_SECONDS),
        )
        self.assertEqual(Freshness(utc, observed, MAX_LIVE_QUOTE_AGE_SECONDS), "fresh")

    def test_source_clock_not_replaced_by_fetch_time_when_missing(self):
        moment, kind = ParseInstant(None)
        self.assertIsNone(moment)
        self.assertEqual(kind, "missing")
        self.assertEqual(Freshness(None, OBSERVED, MAX_LIVE_QUOTE_AGE_SECONDS), "unknown")


if __name__ == "__main__":
    unittest.main()
