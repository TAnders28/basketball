"""Unit tests for in-play WP parsers, leak-free walk-forward, and as-of alignment."""
import math
import unittest

from InPlay.WinProb import (
    DevigMl, FeatureRow, FitGbm, MatrixFromRows, McLookup, PhaseOf, PredictGbm,
    SimulateCell, TimeBin, WalkForwardGbm, WalkPlayState, IsLastFt, IsTeamFoul,
    EstimateRates, REG_END,
)
import numpy as np


def Play(seq, period, clock, hs, as_, typ, team, scoring=False, sv=0, shoot=False, wall=None):
    return {
        "id": str(seq), "seq": seq, "period": period, "clock": clock,
        "hs": hs, "as": as_, "wall": wall or f"2025-10-22T00:00:{seq:02d}Z",
        "type": typ, "team": team, "scoring": scoring, "sv": sv, "shoot": shoot,
    }


def TinyGame(plays, home_won=True):
    return {
        "id": "g1", "date": "2025-10-22T23:00:00Z", "season": 2026, "season_type": 2,
        "home": "NYK", "away": "BOS", "home_id": "18", "away_id": "5",
        "home_won": home_won, "plays": plays, "wp": [], "pickcenter": [],
    }


class ParserTests(unittest.TestCase):
    def test_devig_symmetric(self):
        p = DevigMl(-110, -110)
        self.assertAlmostEqual(p, 0.5, places=5)

    def test_devig_favorite(self):
        p = DevigMl(-200, 170)
        self.assertGreater(p, 0.6)
        self.assertLess(p, 0.75)

    def test_last_ft(self):
        self.assertTrue(IsLastFt("Free Throw - 2 of 2"))
        self.assertFalse(IsLastFt("Free Throw - 1 of 2"))
        self.assertTrue(IsLastFt("Free Throw - 1 of 1"))

    def test_team_fouls_ignore_offensive(self):
        self.assertTrue(IsTeamFoul("Shooting Foul"))
        self.assertTrue(IsTeamFoul("Personal Foul"))
        self.assertFalse(IsTeamFoul("Offensive Foul"))
        self.assertFalse(IsTeamFoul("Offensive Foul Turnover"))

    def test_possession_make_then_rebound(self):
        # Home (18) jumpball, away (5) scores, home misses, away defensive rebound.
        plays = [
            Play(1, 1, "12:00", 0, 0, "Jumpball", "18"),
            Play(2, 1, "11:40", 0, 2, "Jump Shot", "5", scoring=True, sv=2, shoot=True),
            Play(3, 1, "11:20", 0, 2, "Jump Shot", "18", scoring=False, sv=0, shoot=True),
            Play(4, 1, "11:19", 0, 2, "Defensive Rebound", "5"),
        ]
        st = WalkPlayState(TinyGame(plays))
        self.assertEqual(st[0]["poss"], "home")          # jumpball winner
        self.assertEqual(st[1]["poss"], "home")          # after away make, home ball
        self.assertEqual(st[3]["poss"], "away")          # defensive rebound by away

    def test_turnover_flips_possession(self):
        plays = [
            Play(1, 1, "12:00", 0, 0, "Jumpball", "18"),
            Play(2, 1, "11:30", 0, 0, "Bad Pass Turnover", "18"),
        ]
        st = WalkPlayState(TinyGame(plays))
        self.assertEqual(st[1]["poss"], "away")

    def test_team_fouls_reset_each_period_and_no_lookahead(self):
        plays = [
            Play(1, 1, "12:00", 0, 0, "Jumpball", "18"),
            Play(2, 1, "11:00", 0, 0, "Personal Foul", "18"),
            Play(3, 1, "10:00", 0, 0, "Personal Foul", "18"),
            Play(4, 2, "12:00", 0, 0, "Jumpball", "5"),
            Play(5, 2, "11:00", 0, 0, "Shooting Foul", "18"),
        ]
        st = WalkPlayState(TinyGame(plays))
        self.assertEqual(st[2]["home_tf"], 2)
        self.assertEqual(st[3]["home_tf"], 0)  # reset at Q2 start, jumpball not a foul
        self.assertEqual(st[4]["home_tf"], 1)
        # Earlier plays must not see later fouls
        self.assertEqual(st[1]["home_tf"], 1)

    def test_bonus_at_five(self):
        plays = [Play(1, 1, "12:00", 0, 0, "Jumpball", "18")]
        for i in range(5):
            plays.append(Play(10 + i, 1, f"{11-i}:00", 0, 0, "Personal Foul", "5"))
        st = WalkPlayState(TinyGame(plays))
        self.assertEqual(st[-2]["away_bonus"], 0)
        self.assertEqual(st[-1]["away_bonus"], 1)
        self.assertEqual(st[-1]["away_tf"], 5)

    def test_timeouts_count_and_official_ignored(self):
        plays = [
            Play(1, 1, "12:00", 0, 0, "Jumpball", "18"),
            Play(2, 1, "8:00", 0, 0, "Full Timeout", "18"),
            Play(3, 1, "6:00", 0, 0, "Official Timeout", "18"),
        ]
        st = WalkPlayState(TinyGame(plays))
        self.assertEqual(st[1]["home_to_used"], 1)
        self.assertEqual(st[2]["home_to_used"], 1)
        self.assertEqual(st[2]["home_to_left"], 6)

    def test_made_last_ft_flips(self):
        plays = [
            Play(1, 1, "12:00", 0, 0, "Jumpball", "18"),
            Play(2, 1, "11:00", 0, 0, "Shooting Foul", "5"),
            Play(3, 1, "11:00", 1, 0, "Free Throw - 1 of 2", "18", scoring=True, sv=1, shoot=True),
            Play(4, 1, "11:00", 2, 0, "Free Throw - 2 of 2", "18", scoring=True, sv=1, shoot=True),
        ]
        st = WalkPlayState(TinyGame(plays))
        self.assertEqual(st[3]["poss"], "away")


class ModelLogicTests(unittest.TestCase):
    def test_phase(self):
        self.assertEqual(PhaseOf(100, 1), "Q1")
        self.assertEqual(PhaseOf(2000, 3), "Q3")
        self.assertEqual(PhaseOf(2800, 4), "last5")  # 80s left
        self.assertEqual(PhaseOf(3000, 5), "OT")

    def test_walkforward_trains_only_earlier_seasons(self):
        def fake(season, gid, y=1, prior=0.6):
            # minimal games with rows + empty states
            row = FeatureRow(
                {"el": 1000, "per": 2, "margin": 4, "poss": "home",
                 "home_bonus": 0, "away_bonus": 0, "home_tf": 1, "away_tf": 0,
                 "home_to_left": 6, "away_to_left": 7, "run": 0, "wp_espn": 0.6, "t": 0},
                prior, "kalshi", y,
            )
            row.update({"game_id": gid, "date": f"{season}-01-01", "home": "AAA",
                        "away": "BBB", "season": season})
            return {"game_id": gid, "date": row["date"], "home": "AAA", "away": "BBB",
                    "season": season, "won": y, "prior": prior, "prior_source": "kalshi",
                    "rows": [row], "states": [], "n_states": 0}

        games = [fake(2018, "a"), fake(2019, "b"), fake(2020, "c")]
        models = WalkForwardGbm(games, params={"n_estimators": 10, "num_leaves": 8,
                                               "min_child_samples": 1, "max_depth": 3})
        self.assertNotIn(2018, models)          # no earlier season to train on
        self.assertIn(2019, models)
        self.assertIn(2020, models)
        self.assertIsNone(games[0].get("p_gbm_rows"))
        self.assertIsNotNone(games[1].get("p_gbm_rows"))

    def test_gbm_init_score_uses_prior(self):
        # Zero state features except prior; predictions should rise with prior.
        rows = []
        for prior, y in [(0.2, 0), (0.3, 0), (0.7, 1), (0.8, 1)] * 20:
            r = FeatureRow(
                {"el": 1000, "per": 2, "margin": 0, "poss": "home",
                 "home_bonus": 0, "away_bonus": 0, "home_tf": 0, "away_tf": 0,
                 "home_to_left": 7, "away_to_left": 7, "run": 0, "wp_espn": prior, "t": 0},
                prior, "kalshi", y,
            )
            # wipe state features so only prior remains
            for k in ("margin", "margin_z", "poss_home", "poss_away", "run", "foul_diff", "to_diff"):
                r[k] = 0.0
            rows.append(r)
        X, y, init = MatrixFromRows(rows)
        booster = FitGbm(X, y, init, params={"n_estimators": 20, "num_leaves": 8,
                                             "min_child_samples": 5, "max_depth": 2})
        pred = PredictGbm(booster, X, init)
        self.assertGreater(pred[2], pred[0])  # 0.7 prior > 0.2 prior

    def test_mc_leader_higher_wp(self):
        rates = {"p_to": 0.13, "p_3att": 0.35, "p_2att": 0.40, "fg2": 0.52, "fg3": 0.36,
                 "orb": 0.27, "p_ft": 0.15, "sec_per_poss": 14.5, "ft_pts": 1.5}
        rng = np.random.default_rng(0)
        grid = SimulateCell(600, True, False, rates, n_sims=20000, ppp_home_edge=0.0, rng=rng)
        self.assertGreater(grid[8], grid[0])
        self.assertGreater(grid[0], grid[-8])
        self.assertGreater(grid[0], 0.45)  # home has the ball, slight edge

    def test_time_bin_no_future(self):
        self.assertEqual(TimeBin(15), 0)
        self.assertEqual(TimeBin(90), 2)
        self.assertLess(TimeBin(200), TimeBin(2000))


if __name__ == "__main__":
    unittest.main()
