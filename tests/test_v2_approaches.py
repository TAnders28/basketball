"""Generic tests run over EVERY registered V2 approach on a synthetic league: interface, finite outputs,
no-lookahead (perturb results on/after day D -> predictions on days <= D unchanged), plus targeted checks."""

import math
import unittest

import numpy as np

import Matchups.V2.Approaches  # noqa: F401  (registers everything)
from Matchups.V2.Base import REGISTRY, PregameView, RosterBook, ScheduleBook, Walk
from Matchups.V2.Diagnostics import PriorSensitivity
from Matchups.V2.Synthetic import MakeLeague, Perturb

GAMES, RECS, TRUTH = MakeLeague(n_days=20, games_per_day=3, seed=11)
CUT = GAMES[len(GAMES) // 2]["game_date"]
MINUTES_ONLY = {"coach_profile"}          # minutes model: predicts rotations, not margins


def Run(name, recs=RECS, **params):
    return Walk(REGISTRY[name](**params), GAMES, records=lambda g: recs[g["game_id"]])


def Comparable(row):
    out = {}
    for k, v in row.items():
        if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
            out[k] = "nan" if math.isnan(float(v)) else round(float(v), 9)
        elif isinstance(v, (str, list, tuple)):
            out[k] = str(v)
    for k in ("margin", "home_win"):        # realized outcome columns come from the games list, not the model
        out.pop(k, None)
    return out


class TestEveryApproach(unittest.TestCase):
    def test_registry_is_complete(self):
        self.assertGreaterEqual(len(REGISTRY), 17)
        for name, cls in REGISTRY.items():
            self.assertTrue(cls.__doc__ or __import__(cls.__module__).__doc__, name)
            self.assertTrue(cls.idea and cls.data_used and cls.covers and cls.priors, name)

    def test_outputs_finite(self):
        for name in REGISTRY:
            with self.subTest(name=name):
                rows = Run(name)
                self.assertEqual(len(rows), len(GAMES))
                if name in MINUTES_ONLY:
                    continue
                for r in rows:
                    self.assertTrue(math.isfinite(r["margin_mean"]) and r["margin_var"] > 0, (name, r))
                    self.assertTrue(0 < r["p_home"] < 1, (name, r))

    def test_no_lookahead(self):
        perturbed = Perturb(RECS, CUT, seed=5)
        for name in REGISTRY:
            with self.subTest(name=name):
                a, b = Run(name), Run(name, perturbed)
                early_a = [Comparable(r) for r in a if r["game_date"] <= CUT]
                early_b = [Comparable(r) for r in b if r["game_date"] <= CUT]
                self.assertEqual(early_a, early_b)
                late_a = [r.get("p_home") for r in a if r["game_date"] > CUT]
                late_b = [r.get("p_home") for r in b if r["game_date"] > CUT]
                if name not in MINUTES_ONLY and name != "tracking_prior":
                    self.assertNotEqual(late_a, late_b, "perturbation should change later predictions")


class TestLeakGuards(unittest.TestCase):
    def test_prior_season_only(self):
        v = PregameView(GAMES[0], RosterBook(), ScheduleBook())
        with self.assertRaises(PermissionError):
            v.Prior("synergy/Isolation_offensive_P.json", season=v.season)


class TestTargeted(unittest.TestCase):
    def test_kalman_learns_strength_order(self):
        games, recs, truth = MakeLeague(n_days=80, games_per_day=3, seed=2)
        a = REGISTRY["kalman_team"]()
        Walk(a, games, records=lambda g: recs[g["game_id"]])
        ix = a.core.ix
        est = {t: a.core.k.x[ix[t]] - a.core.k.x[30 + ix[t]] for t in truth["team_strength"]}
        r = np.corrcoef([est[t] for t in est], [truth["team_strength"][t] for t in est])[0, 1]
        self.assertGreater(r, 0.6)

    def test_prior_sensitivity_kalman(self):
        def Fn(scale):
            return [r["p_home"] for r in Run("kalman_team", prior_var=16.0 * scale)]
        s = PriorSensitivity(Fn)
        self.assertGreater(s[2.0]["max_abs"], 0)
        self.assertLess(s[2.0]["mean_abs"], 0.15)

    def test_prior_sensitivity_residual_family(self):
        def Fn(scale):
            return [r["p_home"] for r in Run("schedule_fatigue", prior_var=4.0 * scale)]
        s = PriorSensitivity(Fn)
        self.assertLess(s[2.0]["mean_abs"], 0.1)

    def test_stacker_refits_and_starts_from_kalman(self):
        rows = Run("stacker", min_rows=20, refit_every=3, members=("kalman_team", "four_factors"))
        self.assertGreater(rows[-1]["stack_rows"], 20)
        self.assertNotEqual(rows[-1]["w_kalman_team"], 1.6)
        self.assertEqual(rows[0]["w_kalman_team"], 1.6)

    def test_lineup_vs_lineup_lists_fives(self):
        rows = Run("lineup_vs_lineup")
        self.assertTrue(rows[-1]["home_fives"] and len(rows[-1]["home_fives"][0]) == 5)
        self.assertGreaterEqual(rows[-1]["minutes_var_term"], 0)

    def test_coach_profile_projects_minutes(self):
        a = REGISTRY["coach_profile"]()
        rows = Walk(a, GAMES, records=lambda g: RECS[g["game_id"]])
        self.assertTrue(a.diagnostics["mae_coach"] and a.diagnostics["mae_base"])
        self.assertTrue(all(math.isfinite(x) for x in a.diagnostics["mae_coach"]))


if __name__ == "__main__":
    unittest.main()
