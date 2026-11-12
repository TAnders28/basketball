import os
import unittest

import numpy as np

from Matchups import Cpu, Scoring
from Matchups.Models import MatchupModel, MinutesBook, RapmModel
from Matchups.WalkForward import Walk


def SyntheticSeason():
    """Four teams, three players each, deterministic results; season 2023 so the walk records rows."""
    teams = {"AAA": [1, 2, 3], "BBB": [4, 5, 6], "CCC": [7, 8, 9], "DDD": [10, 11, 12]}
    games, players, poss = [], {}, {}
    schedule = [("AAA", "BBB"), ("CCC", "DDD"), ("AAA", "CCC"), ("BBB", "DDD"), ("AAA", "DDD"), ("BBB", "CCC")] * 3
    for i, (h, a) in enumerate(schedule):
        gid = f"00223{i:05d}"
        day = f"2023-11-{1 + i // 2:02d}"
        hs, as_ = 110 + (5 if h == "AAA" else 0), 105 + (i % 3)
        games.append({"game_id": gid, "game_date": day, "season": 2023, "home_team": h, "away_team": a,
                      "home_score": hs, "away_score": as_})
        players[gid] = [{"player_id": p, "team": t, "minutes": 80.0, "pts": 30, "plus_minus": 0, "name": f"P{p}"}
                        for t in (h, a) for p in teams[t]]
        poss[(gid, h)] = poss[(gid, a)] = 100.0
    return games, players, poss


class CpuCapTests(unittest.TestCase):
    def test_workers_at_most_half_the_cores(self):
        cap = max(1, (os.cpu_count() or 2) // 2)
        self.assertEqual(Cpu.MaxWorkers(), cap)
        self.assertEqual(Cpu.MaxWorkers(10_000), cap)
        self.assertEqual(Cpu.MaxWorkers(1), 1)

    def test_thread_env(self):
        Cpu.LimitThreads()
        for var in Cpu.THREAD_VARS:
            self.assertEqual(os.environ[var], "1")


class NoLookaheadTests(unittest.TestCase):
    def test_changing_a_result_never_changes_that_days_or_earlier_predictions(self):
        games, players, poss = SyntheticSeason()
        base = Walk({"model": "rapm", "lam": 10, "half_life": 120, "name": "t"}, data=(games, players, poss))
        cut = games[8]["game_date"]
        altered = [dict(g) for g in games]
        for g in altered:
            if g["game_date"] >= cut:
                g["home_score"], g["away_score"] = 80, 140
        alt = Walk({"model": "rapm", "lam": 10, "half_life": 120, "name": "t"}, data=(altered, players, poss))
        for r0, r1 in zip(base["rows"], alt["rows"]):
            if r0["game_date"] <= cut:
                self.assertAlmostEqual(r0["pred_margin_pregame"], r1["pred_margin_pregame"])
        later = [(a["pred_margin_pregame"], b["pred_margin_pregame"]) for a, b in zip(base["rows"], alt["rows"]) if a["game_date"] > cut]
        self.assertTrue(any(abs(a - b) > 1e-6 for a, b in later))


class ModelTests(unittest.TestCase):
    def test_matchup_model_finds_better_scorer_and_stopper(self):
        model = MatchupModel(k=2.0, half_life_days=365)
        rows = []
        for _ in range(50):
            rows += [{"off": 1, "def": 10, "poss": 10, "pts": 4}, {"off": 1, "def": 11, "poss": 10, "pts": 1},
                     {"off": 2, "def": 10, "poss": 10, "pts": 2}, {"off": 2, "def": 11, "poss": 10, "pts": 1}]
        model.AddRows(rows)
        model.Fit(iterations=20)
        self.assertGreater(model.OffenseRate(1), model.OffenseRate(2))
        self.assertLess(model.Lambda(1, 11), model.Lambda(1, 10))
        shares = model.Shares(1, {10: 30.0, 11: 30.0, 12: 30.0})
        self.assertAlmostEqual(sum(shares.values()), 1.0)
        self.assertLess(shares[12], shares[10])   # never-seen defender gets only prior weight

    def test_rapm_recovers_sign(self):
        model = RapmModel(lam=1.0, half_life_days=1000)
        for i in range(40):
            g = {"home_team": "H", "away_team": "A", "home_score": 110, "away_score": 100}
            rows = [{"player_id": 1 if i % 2 else 3, "team": "H", "minutes": 48}, {"player_id": 2, "team": "A", "minutes": 48}]
            model.Apply(g, rows)
        model.Fit()
        self.assertGreater(model.Rating(1) + model.Rating(3) + 2 * model.ratings[0], 0)

    def test_minutes_book_scales_to_240(self):
        book = MinutesBook()
        roster = book.OracleRoster([1, 2, 3])
        self.assertAlmostEqual(sum(roster.values()), 240.0)


class ScoringTests(unittest.TestCase):
    def test_logistic_recovers_slope(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=4000)
        y = (rng.random(4000) < Scoring.Sigmoid(0.3 + 1.5 * x)).astype(float)
        w = Scoring.FitLogistic(x[:, None], y)
        self.assertAlmostEqual(w[1], 1.5, delta=0.2)

    def test_bootstrap_identical_is_zero(self):
        p = np.full(100, 0.6)
        y = np.ones(100)
        out = Scoring.BootstrapDiff(p, p, y, [i // 5 for i in range(100)], reps=200)
        self.assertEqual(out["d_log_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()
