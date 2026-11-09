"""Holdout protocol smoke tests. Uses tiny synthetic seasons; no network."""

import json
import os
import tempfile
import unittest
from unittest import mock

from Backtest import RunBacktest
from Backtest.Strategies import DefaultStrategies


def FakeGame(gid, day, home, away, hs, aws):
    return {
        "game_id": gid, "game_date": day, "season_type": "Regular Season",
        "home_team": home, "away_team": away, "home_score": hs, "away_score": aws,
    }


class HoldoutTests(unittest.TestCase):
    def test_settings_grid_count_reported(self):
        grid = RunBacktest.SettingsGrid()
        self.assertEqual(len(grid), 3 * 2 * 3)

    def test_holdout_selects_and_scores_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            season_dir = os.path.join(tmp, "season")
            report_dir = os.path.join(tmp, "reports")
            os.makedirs(season_dir)
            # Two short tune seasons + holdout
            for year, offset in ((2022, 0), (2023, 1), (2024, 2)):
                games = []
                for i in range(6):
                    day = f"{year + 1}-01-{i + 1:02d}"
                    games.append(FakeGame(f"{year}-{i}", day, "AAA", "BBB", 110 - i, 100))
                    games.append(FakeGame(f"{year}-b{i}", day, "CCC", "DDD", 105, 100 + i))
                with open(os.path.join(season_dir, f"games_{year}.json"), "w", encoding="utf-8") as fh:
                    json.dump(games, fh)
            with mock.patch.object(RunBacktest, "SEASON_DIR", season_dir), \
                 mock.patch.object(RunBacktest, "REPORT_DIR", report_dir), \
                 mock.patch.object(RunBacktest, "SettingsGrid",
                                   return_value=[{"min_games": 3, "min_gap": 1, "min_edge": 0.05},
                                                 {"min_games": 5, "min_gap": 1, "min_edge": 0.05}]):
                payload = RunBacktest.RunHoldout([2022, 2023], 2024)
            self.assertEqual(payload["settings_tried"], 2)
            self.assertIn("selected_settings", payload)
            self.assertEqual(payload["holdout_season"], 2024)
            self.assertEqual(payload["n_per_season"]["2024"], 12)
            self.assertTrue(os.path.exists(os.path.join(report_dir, "holdout_2024.json")))
            self.assertEqual(payload["worst_season_rank_on_holdout"][0]["rank_worst_first"], 1)

    def test_default_mode_unchanged_api(self):
        # DefaultStrategies still the default path for RunOne
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = os.path.join(tmp, "reports")
            with mock.patch.object(RunBacktest, "REPORT_DIR", report_dir):
                games = [FakeGame("1", "2026-01-01", "AAA", "BBB", 110, 100)]
                result = RunBacktest.RunOne("sample", games, [])
            self.assertIn("elo_pick", result["strategies"])
            self.assertEqual(len(DefaultStrategies()), 5)


if __name__ == "__main__":
    unittest.main()
