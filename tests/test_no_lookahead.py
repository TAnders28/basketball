"""Leak / look-ahead guards. Synthetic fixtures only — no network."""

import copy
import unittest

from Backtest.Engine import PreGameView, RunSeason
from Backtest.Features import FeatureBook
from Backtest.Model import EloModel
from Backtest.Strategies import DefaultStrategies, EloPick


def FakeGame(game_id, game_date, home, away, home_score, away_score):
    return {
        "game_id": game_id,
        "game_date": game_date,
        "season_type": "Regular Season",
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
    }


class NoLookaheadTests(unittest.TestCase):
    def test_run_season_same_day_games_do_not_see_each_other(self):
        """Both games on D are decided before either result is Applied."""
        early = [
            FakeGame("1", "2026-01-01", "AAA", "BBB", 110, 100),
            FakeGame("2", "2026-01-01", "CCC", "DDD", 90, 95),
        ]
        books = RunSeason(early, [EloPick()])
        picks = {p["game_id"]: p for p in books["elo_pick"]["picks"]}
        # With no history both sides start at base Elo; home court -> home fav.
        self.assertEqual(picks["1"]["pick"], "home")
        self.assertEqual(picks["2"]["pick"], "home")
        # Flipping the other same-day result must not change the first game's pick or model_prob.
        flipped = [
            FakeGame("1", "2026-01-01", "AAA", "BBB", 110, 100),
            FakeGame("2", "2026-01-01", "CCC", "DDD", 120, 80),  # different result
        ]
        books2 = RunSeason(flipped, [EloPick()])
        picks2 = {p["game_id"]: p for p in books2["elo_pick"]["picks"]}
        self.assertEqual(picks["1"]["pick"], picks2["1"]["pick"])
        self.assertEqual(picks["1"]["model_prob"], picks2["1"]["model_prob"])

    def test_future_games_cannot_change_earlier_day_picks(self):
        day1 = [
            FakeGame("a", "2026-01-01", "AAA", "BBB", 110, 100),
            FakeGame("b", "2026-01-01", "CCC", "DDD", 105, 100),
        ]
        with_future = day1 + [
            FakeGame("z", "2026-01-10", "AAA", "CCC", 50, 150),  # huge blowout later
        ]
        books_short = RunSeason(day1, DefaultStrategies())
        books_long = RunSeason(with_future, DefaultStrategies())
        for name in books_short:
            short_by_id = {p["game_id"]: p for p in books_short[name]["picks"] if p["game_date"] == "2026-01-01"}
            long_by_id = {p["game_id"]: p for p in books_long[name]["picks"] if p["game_date"] == "2026-01-01"}
            self.assertEqual(set(short_by_id), set(long_by_id), name)
            for gid, row in short_by_id.items():
                self.assertEqual(row["pick"], long_by_id[gid]["pick"], f"{name} {gid}")
                self.assertEqual(row["model_prob"], long_by_id[gid]["model_prob"], f"{name} {gid}")

    def test_predictday_frozen_context_ignores_same_day_and_future(self):
        """Mirrors PredictDay.BuildState: only completed games with date < target are Applied."""
        history = [
            FakeGame("1", "2026-01-01", "AAA", "BBB", 110, 100),
            FakeGame("2", "2026-01-02", "AAA", "CCC", 100, 90),
        ]
        same_day = FakeGame("3", "2026-01-03", "AAA", "DDD", 80, 120)
        future = FakeGame("4", "2026-01-04", "BBB", "CCC", 130, 70)
        target = "2026-01-03"

        def Freeze(games, target_date):
            features, model = FeatureBook(), EloModel()
            played = sorted(
                (g for g in games if g["home_score"] is not None and g["game_date"] < target_date),
                key=lambda g: (g["game_date"], g["game_id"]),
            )
            for game in played:
                features.Apply(game)
                model.Apply(game)
            return features, model, played

        f1, m1, p1 = Freeze(history, target)
        f2, m2, p2 = Freeze(history + [same_day, future], target)
        self.assertEqual([g["game_id"] for g in p1], [g["game_id"] for g in p2])
        self.assertEqual(m1.Rating("AAA"), m2.Rating("AAA"))
        self.assertEqual(m1.Rating("BBB"), m2.Rating("BBB"))
        slate = {"game_id": "slate", "game_date": target, "home_team": "AAA", "away_team": "BBB",
                 "season_type": "Regular Season"}
        v1 = PreGameView(slate, f1, m1)
        v2 = PreGameView(slate, f2, m2)
        self.assertEqual(v1["model_home_prob"], v2["model_home_prob"])
        self.assertEqual(v1["home_win_pct"], v2["home_win_pct"])
        # Control: including same-day *as if earlier* would change ratings — proves the date cut matters.
        f3, m3, _ = Freeze(history + [{**same_day, "game_date": "2026-01-02"}], target)
        self.assertNotEqual(m1.Rating("AAA"), m3.Rating("AAA"))


if __name__ == "__main__":
    unittest.main()
