"""Unit tests for leak-free four-factor GAP ratings."""
import math
import unittest

from Backtest.FourFactorsGap import FactorsFromBoxes, FourFactorsGap, Ppp, WalkRatings


def Box(fgm, fga, fg3m, ftm, fta, oreb, dreb, tov, minutes=240):
    return {"FGM": fgm, "FGA": fga, "FG3M": fg3m, "FTM": ftm, "FTA": fta,
            "OREB": oreb, "DREB": dreb, "TOV": tov, "MIN": minutes}


class FourFactorsGapTests(unittest.TestCase):
    def test_factor_formulas(self):
        team = Box(40, 80, 10, 16, 20, 10, 30, 12)
        opp = Box(35, 85, 8, 10, 14, 8, 30, 14)
        f = FactorsFromBoxes(team, opp)
        self.assertAlmostEqual(f["efg"][0] / f["efg"][1], (40 + 5) / 80)
        self.assertAlmostEqual(f["tov"][0] / f["tov"][1], 12 / (80 + 0.44 * 20 + 12))
        self.assertAlmostEqual(f["orb"][0] / f["orb"][1], 10 / (10 + 30))
        self.assertAlmostEqual(f["ftr"][0] / f["ftr"][1], 20 / 80)
        self.assertAlmostEqual(f["ft"][0] / f["ft"][1], 16 / 20)

    def test_ppp_increases_with_efg(self):
        low = Ppp(0.45, 0.13, 0.27, 0.26, 0.77)
        high = Ppp(0.60, 0.13, 0.27, 0.26, 0.77)
        self.assertGreater(high, low)

    def test_ppp_falls_with_tov(self):
        self.assertGreater(Ppp(0.54, 0.10, 0.27, 0.26, 0.77), Ppp(0.54, 0.20, 0.27, 0.26, 0.77))

    def test_same_day_does_not_leak(self):
        # Two games on the same day involving HOU. The second game must not
        # see the first game's box scores.
        model = FourFactorsGap()
        model.BeforeDay("2025-10-21", 0)
        p1 = model.Predict("HOU", "OKC")
        # Do not Update yet; a second same-day prediction must match a fresh prior.
        p2 = model.Predict("HOU", "LAL")
        model2 = FourFactorsGap()
        model2.BeforeDay("2025-10-21", 0)
        p2_fresh = model2.Predict("HOU", "LAL")
        self.assertAlmostEqual(p2["p_home"], p2_fresh["p_home"])
        # After an update, the next day moves off the prior.
        model.Update("HOU", "OKC",
                     Box(50, 80, 15, 20, 24, 12, 32, 8),
                     Box(30, 90, 5, 8, 12, 6, 20, 18))
        model.BeforeDay("2025-10-22", 1)
        p_next = model.Predict("HOU", "LAL")
        self.assertNotAlmostEqual(p_next["p_home"], p2["p_home"], places=6)

    def test_walk_updates_only_after_the_day(self):
        games = [
            {"game_id": "a", "game_date": "2025-10-21", "home": "AAA", "away": "BBB",
             "home_box": Box(50, 80, 15, 20, 24, 12, 32, 8),
             "away_box": Box(30, 90, 5, 8, 12, 6, 20, 18)},
            {"game_id": "b", "game_date": "2025-10-21", "home": "AAA", "away": "CCC",
             "home_box": Box(40, 80, 10, 10, 12, 8, 30, 12),
             "away_box": Box(40, 80, 10, 10, 12, 8, 30, 12)},
            {"game_id": "c", "game_date": "2025-10-22", "home": "AAA", "away": "DDD",
             "home_box": Box(40, 80, 10, 10, 12, 8, 30, 12),
             "away_box": Box(40, 80, 10, 10, 12, 8, 30, 12)},
        ]
        out = WalkRatings(games)
        # a and b are same-day, so they share the opening prior (AAA vs different
        # opponents still start from the same league prior, so p_home should be
        # identical at the prior). c must differ after the day-21 update.
        self.assertAlmostEqual(out["a"]["p_home"], out["b"]["p_home"])
        self.assertNotAlmostEqual(out["c"]["p_home"], out["a"]["p_home"], places=6)

    def test_good_offense_raises_p_home(self):
        # After AAA blows out BBB, AAA at home vs a prior-level opponent should
        # be a bigger favorite than at the opening prior.
        prior_games = [{
            "game_id": "w1", "game_date": "2025-10-21", "home": "AAA", "away": "BBB",
            "home_box": Box(55, 85, 18, 22, 25, 14, 36, 7),
            "away_box": Box(28, 90, 4, 6, 10, 5, 18, 20),
        }]
        probe = [{
            "game_id": "w2", "game_date": "2025-10-23", "home": "AAA", "away": "CCC",
            "home_box": Box(40, 80, 10, 10, 12, 8, 30, 12),
            "away_box": Box(40, 80, 10, 10, 12, 8, 30, 12),
        }]
        out = WalkRatings(prior_games + probe)
        opening = WalkRatings(probe)
        self.assertGreater(out["w2"]["p_home"], opening["w2"]["p_home"])

    def test_probabilities_in_unit_interval(self):
        model = FourFactorsGap()
        p = model.Predict("HOU", "OKC")["p_home"]
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)


if __name__ == "__main__":
    unittest.main()