"""Tests for append-only prediction log and grader. No network."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest import mock
from zoneinfo import ZoneInfo

from Backtest import GradePredictions, PredictionLog

EASTERN = ZoneInfo("America/New_York")


class PredictionLogTests(unittest.TestCase):
    def test_append_never_rewrites_and_null_markets(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(PredictionLog, "PRED_LOG_DIR", tmp):
                rows = [{
                    "game_id": "g1", "game_date": "2026-11-08", "tip_et": "7:00 pm ET",
                    "home_team": "MIN", "away_team": "UTA",
                    "recommended": "MIN", "recommended_prob": 0.9,
                    "market_price": None, "price_time": None,
                }]
                path = PredictionLog.AppendPredictionRows(2026, rows, created_at="2026-11-08T12:00:00-05:00")
                PredictionLog.AppendPredictionRows(2026, rows, created_at="2026-11-08T12:05:00-05:00")
                with open(path, encoding="utf-8") as fh:
                    lines = fh.read().strip().splitlines()
                self.assertEqual(len(lines), 2)
                first, second = json.loads(lines[0]), json.loads(lines[1])
                self.assertEqual(first["created_at"], "2026-11-08T12:00:00-05:00")
                self.assertEqual(second["created_at"], "2026-11-08T12:05:00-05:00")
                self.assertIsNone(first["market_price"])
                self.assertIsNone(first["price_time"])
                self.assertEqual(first["model_version"], PredictionLog.MODEL_VERSION)
                self.assertTrue(first["start_time"].startswith("2026-11-08T19:00:00"))

    def test_grader_excludes_created_after_start_and_keys_by_game_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            season_dir = os.path.join(tmp, "season")
            os.makedirs(season_dir)
            games = [{
                "game_id": "g1", "game_date": "2026-11-08",
                "home_team": "MIN", "away_team": "UTA",
                "home_score": 115, "away_score": 100,
            }]
            with open(os.path.join(season_dir, "games_2026.json"), "w", encoding="utf-8") as fh:
                json.dump(games, fh)
            with mock.patch.object(PredictionLog, "PRED_LOG_DIR", tmp), \
                 mock.patch.object(GradePredictions, "SEASON_DIR", season_dir):
                # eligible: created before tip
                PredictionLog.AppendPredictionRows(2026, [{
                    "game_id": "g1", "game_date": "2026-11-08", "tip_et": "7:00 pm ET",
                    "home_team": "MIN", "away_team": "UTA",
                    "recommended": "MIN", "recommended_prob": 0.9,
                }], created_at="2026-11-08T12:00:00-05:00")
                # excluded: created after tip
                PredictionLog.AppendPredictionRows(2026, [{
                    "game_id": "g1", "game_date": "2026-11-08", "tip_et": "7:00 pm ET",
                    "home_team": "MIN", "away_team": "UTA",
                    "recommended": "UTA", "recommended_prob": 0.2,
                }], created_at="2026-11-08T20:00:00-05:00")
                summary = GradePredictions.GradeSeason(2026)
                self.assertEqual(summary["graded"], 1)
                self.assertTrue(summary["grades"]["g1"]["correct"])
                self.assertEqual(summary["grades"]["g1"]["pick"], "MIN")
                self.assertEqual(len(summary["excluded_rows"]), 1)
                self.assertEqual(summary["excluded_rows"][0]["reason"], "created_after_start")
                self.assertTrue(os.path.exists(PredictionLog.GradesPath(2026)))


if __name__ == "__main__":
    unittest.main()

