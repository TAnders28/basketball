"""Unit tests for the leak-free 3PM props model and Gaussian copula joint."""
import math
import unittest

from Props.Model import (
    AttachPredictions, BuildFeatures, EmpiricalBayesK, EstimateRho, FilterCalib,
    FitParams, FitPoissonOffset, GaussianCopulaC, JointGe, MidPit, NbGe,
    ParseMatchup, PoissonGe, PregameRow, RangeRow, Shrink, ApplyOffset,
)


def P(pid, name, team, date, gid, mins, m3, a3, matchup, season="22024"):
    return {
        "SEASON_ID": season, "PLAYER_ID": pid, "PLAYER_NAME": name,
        "TEAM_ID": 1, "TEAM_ABBREVIATION": team, "TEAM_NAME": team,
        "GAME_ID": gid, "GAME_DATE": date, "MATCHUP": matchup,
        "MIN": mins, "FG3M": m3, "FG3A": a3,
        "FGM": 0, "FGA": 0, "OREB": 0, "TOV": 0, "FTA": 0,
    }


def G(gid, date, home, away, h3a=30, a3a=30, hfga=90, afga=90):
    def st(pts, fga, t3):
        return {"fgm": 40, "fga": fga, "fg3m": t3 // 3, "fg3a": t3,
                "ftm": 20, "fta": 25, "oreb": 10, "tov": 14,
                "pts": pts, "min": 240}
    return {
        "game_id": gid, "season": 2024, "season_type": "Regular Season",
        "game_date": date, "home_team": home, "away_team": away,
        "home_team_name": home, "away_team_name": away,
        "home_score": 110, "away_score": 105, "home_win": True,
        "home_stats": st(110, hfga, h3a), "away_stats": st(105, afga, a3a),
    }


class ParseAndMathTests(unittest.TestCase):
    def test_matchup_home_away(self):
        t, o, h = ParseMatchup("GSW vs. OKC")
        self.assertEqual((t, o, h), ("GSW", "OKC", True))
        t, o, h = ParseMatchup("GSW @ LAL")
        self.assertEqual((t, o, h), ("GSW", "LAL", False))

    def test_shrink_to_mean_with_no_data(self):
        self.assertAlmostEqual(Shrink(0, 0, 0.36, 200), 0.36, places=6)

    def test_shrink_pulls_extreme_low_sample(self):
        raw = 8 / 10
        shrunk = Shrink(8, 10, 0.36, 200)
        self.assertLess(abs(shrunk - 0.36), abs(raw - 0.36))
        self.assertGreater(shrunk, 0.36)
        self.assertLess(shrunk, raw)

    def test_poisson_ge_monotone_and_matches_pmf(self):
        lam = 2.0
        prev = 1.0
        acc = 0.0
        from scipy.stats import poisson
        for k in range(0, 8):
            p = PoissonGe(k, lam)
            self.assertLessEqual(p, prev + 1e-12)
            prev = p
            if k >= 1:
                acc += float(poisson.pmf(k - 1, lam))
                self.assertAlmostEqual(p, 1.0 - acc, places=8)

    def test_nb_overdispersed_vs_poisson(self):
        lam, alpha, k = 2.0, 0.4, 5
        self.assertGreater(NbGe(k, lam, alpha), PoissonGe(k, lam))
        self.assertAlmostEqual(NbGe(k, lam, 0.0), PoissonGe(k, lam), places=8)

    def test_copula_independence_and_positive_rho(self):
        p_ind = JointGe(4, 4, 2.0, 2.0, 0.2, 0.0, family="nb")
        p_pos = JointGe(4, 4, 2.0, 2.0, 0.2, 0.35, family="nb")
        p_x = NbGe(4, 2.0, 0.2)
        p_y = NbGe(4, 2.0, 0.2)
        self.assertAlmostEqual(p_ind, p_x * p_y, places=5)
        self.assertGreater(p_pos, p_ind)
        self.assertAlmostEqual(GaussianCopulaC(0.3, 0.4, 0.0), 0.12, places=6)


class LeakFreeTests(unittest.TestCase):
    def test_features_exclude_same_and_future_games(self):
        players = [
            P(1, "A", "GSW", "2024-11-01", "0022400001", 30, 3, 7, "GSW vs. LAL"),
            P(1, "A", "GSW", "2024-11-03", "0022400002", 28, 1, 4, "GSW @ OKC"),
            P(1, "A", "GSW", "2024-11-06", "0022400003", 32, 4, 8, "GSW vs. BOS"),
            P(2, "B", "LAL", "2024-11-01", "0022400001", 20, 1, 3, "LAL @ GSW"),
        ]
        games = [
            G("0022400001", "2024-11-01", "GSW", "LAL"),
            G("0022400002", "2024-11-03", "OKC", "GSW"),
            G("0022400003", "2024-11-06", "GSW", "BOS"),
        ]
        rows, pst, tst = BuildFeatures(players, games)
        r1 = next(r for r in rows if r["game_id"] == "0022400001" and r["player_id"] == 1)
        r2 = next(r for r in rows if r["game_id"] == "0022400002" and r["player_id"] == 1)
        r3 = next(r for r in rows if r["game_id"] == "0022400003" and r["player_id"] == 1)
        self.assertEqual(r1["prior_n"], 0)
        self.assertEqual(r1["prior_3pm"], 0)
        self.assertIsNone(r1["proj_min_raw"])
        self.assertEqual(r2["prior_n"], 1)
        self.assertEqual(r2["prior_min"], 30)
        self.assertEqual(r2["prior_3pm"], 3)
        self.assertEqual(r2["prior_3pa"], 7)
        self.assertAlmostEqual(r2["proj_min_raw"], 30)
        self.assertFalse(r2["b2b"])
        self.assertEqual(r3["prior_n"], 2)
        self.assertEqual(r3["prior_min"], 58)
        self.assertEqual(r3["prior_3pm"], 4)
        self.assertEqual(r3["prior_3pa"], 11)
        self.assertEqual(pst[1]["n"], 3)
        self.assertEqual(pst[1]["fg3m"], 8)

    def test_b2b_and_home_flags(self):
        players = [
            P(1, "A", "GSW", "2024-11-01", "0022400001", 30, 2, 5, "GSW vs. LAL"),
            P(1, "A", "GSW", "2024-11-02", "0022400002", 24, 1, 4, "GSW @ SAC"),
        ]
        games = [
            G("0022400001", "2024-11-01", "GSW", "LAL"),
            G("0022400002", "2024-11-02", "SAC", "GSW"),
        ]
        rows, _, _ = BuildFeatures(players, games)
        r2 = next(r for r in rows if r["game_id"] == "0022400002")
        self.assertTrue(r2["b2b"])
        self.assertEqual(r2["rest_days"], 1)
        self.assertFalse(r2["home"])

    def test_train_params_do_not_use_holdout_rows(self):
        rows_p, games = [], []
        for i, d in enumerate(["2024-11-01", "2024-11-03", "2024-11-05", "2024-11-08",
                               "2024-11-10", "2024-11-12", "2024-11-14", "2024-11-16"]):
            gid = f"0022401{i:03d}"
            rows_p.append(P(1, "A", "GSW", d, gid, 30, 2, 6, "GSW vs. LAL", "22024"))
            games.append(G(gid, d, "GSW", "LAL"))
        for i, d in enumerate(["2025-11-01", "2025-11-03"]):
            gid = f"0022501{i:03d}"
            rows_p.append(P(1, "A", "GSW", d, gid, 40, 8, 12, "GSW vs. LAL", "22025"))
            games.append(G(gid, d, "GSW", "LAL", h3a=80))
        feats, _, _ = BuildFeatures(rows_p, games)
        train = [r for r in feats if r["season"] == 2024]
        hold = [r for r in feats if r["season"] == 2025]
        self.assertTrue(train)
        self.assertTrue(hold)
        h0 = hold[0]
        self.assertEqual(h0["prior_n"], 8)
        self.assertEqual(h0["prior_3pm"], 16)
        self.assertLess(h0["prior_3pa"], 60)
        params = FitParams(train)
        self.assertIn("alpha", params)
        self.assertEqual(len(params["beta"]), 5)

    def test_attach_and_range(self):
        row = {
            "player_id": 1, "player_name": "A", "team": "GSW", "opp": "LAL",
            "game_id": "x", "date": "2026-11-13", "season": 2026, "home": True,
            "rest_days": 2, "b2b": False, "actual_min": None, "actual_3pm": None,
            "actual_3pa": None, "prior_n": 12, "prior_min": 360.0, "prior_3pm": 24.0,
            "prior_3pa": 60.0, "proj_min_raw": 30.0, "pace": 100.0, "opp_3pa100": 35.0,
            "league_pace": 100.0, "league_opp3": 35.0,
        }
        params = {
            "mu_pct": 0.36, "k_pct": 200.0, "mu_rate": 0.12, "k_rate": 200.0,
            "beta": [0.0, 0.0, 0.0, 0.0, 0.0], "alpha": 0.2,
            "league_pace": 100.0, "league_opp3": 35.0, "pace_sd": 3.0, "opp3_sd": 4.0,
        }
        AttachPredictions([row], params)
        self.assertGreater(row["lam"], 0)
        self.assertGreater(row["p_nb_2"], row["p_nb_5"])
        rng = RangeRow(row, params, 24.0, 36.0, k_mult_lo=2.0, k_mult_hi=0.5)
        self.assertLess(rng["lam_lo"], rng["lam_hi"])


if __name__ == "__main__":
    unittest.main()
