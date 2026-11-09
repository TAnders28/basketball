"""Scoreboard hardening and Line column for PredictDay. No network."""

import unittest
from unittest import mock

from Backtest import PredictDay


class FakeEndpointOk:
    @staticmethod
    def ScoreboardV2(game_date, timeout=60):
        class R:
            def get_dict(self):
                return {
                    "resultSets": [
                        {"name": "GameHeader", "headers": ["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID",
                                                           "GAMECODE", "GAME_STATUS_TEXT"],
                         "rowSet": [["0022600001", 1, 2, "20261108/UTAMIN", "7:00 pm ET"]]},
                        {"name": "LineScore", "headers": ["TEAM_ID", "TEAM_ABBREVIATION"],
                         "rowSet": [[1, "MIN"], [2, "UTA"]]},
                    ]
                }
        return R()


class FakeEndpointEmpty:
    @staticmethod
    def ScoreboardV2(game_date, timeout=60):
        class R:
            def get_dict(self):
                return {}  # empty / non-JSON shape
        return R()


class PredictDayScoreboardTests(unittest.TestCase):
    def test_format_line_dash_without_odds(self):
        self.assertEqual(PredictDay.FormatLine({}), "-")
        self.assertEqual(PredictDay.FormatLine({"home_moneyline": -110, "away_moneyline": None}), "-")
        self.assertEqual(PredictDay.FormatLine({"home_moneyline": -150, "away_moneyline": 130}), "130/-150")

    def test_scoreboard_sets_retries_then_raises_on_empty(self):
        with mock.patch.object(PredictDay, "SCOREBOARD_RETRY_SLEEP", 0):
            with self.assertRaises(RuntimeError) as ctx:
                PredictDay.ScoreboardSets(FakeEndpointEmpty, "2026-11-08", retries=2)
            self.assertIn("ScoreboardV2 failed", str(ctx.exception))

    def test_fetch_slate_falls_back_to_cdn_on_empty_scoreboard(self):
        cdn_body = {
            "scoreboard": {
                "gameDate": "2026-11-08",
                "games": [{
                    "gameId": "0022600099",
                    "gameStatusText": "Final",
                    "homeTeam": {"teamTricode": "MIN"},
                    "awayTeam": {"teamTricode": "UTA"},
                }],
            }
        }
        with mock.patch.object(PredictDay, "SCOREBOARD_RETRY_SLEEP", 0), \
             mock.patch("nba_api.stats.endpoints.scoreboardv2", FakeEndpointEmpty), \
             mock.patch("LiveOdds.Sources.FetchScoreboard",
                        return_value={"body": cdn_body, "error": None}), \
             mock.patch("LiveOdds.Sources.ReadOnlyHttp"):
            slate = PredictDay.FetchSlate("2026-11-08")
        self.assertEqual(len(slate), 1)
        self.assertEqual(slate[0]["home_team"], "MIN")
        self.assertEqual(slate[0]["line"], "-")
        self.assertEqual(slate[0]["slate_source"], "cdn_scoreboard")

    def test_fetch_slate_happy_path(self):
        with mock.patch("nba_api.stats.endpoints.scoreboardv2", FakeEndpointOk):
            slate = PredictDay.FetchSlate("2026-11-08")
        self.assertEqual(slate[0]["home_team"], "MIN")
        self.assertEqual(slate[0]["away_team"], "UTA")
        self.assertEqual(slate[0]["line"], "-")


if __name__ == "__main__":
    unittest.main()
