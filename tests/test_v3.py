"""V3 tests: evaluation harness, overfitting estimates, trial registry, post-processing (EMOS, temperature,
context stacking) with no-lookahead checks, availability scenarios, learned prior variance, skills, player Kalman."""

import math
import os
import tempfile
import unittest

import numpy as np

import Matchups.V2.Approaches  # noqa: F401
from Matchups.V2.Base import REGISTRY, Walk
from Matchups.V2.Bayes import BayesLinear
from Matchups.V2.Synthetic import MakeLeague
from Matchups.V3 import Evaluate as E
from Matchups.V3 import Overfit as O
from Matchups.V3 import Post as PP
from Matchups.V3 import Trials as T
from Matchups.V3.Availability import Mixture, StatusModel
from Matchups.V2.Approaches.Common import LearnPriorScale
from Matchups.V3.Skills import SkillBook


def FakeRows(n_days=200, per_day=6, sd_scale=1.0, bias=0.0, seed=0, noise_model=0.0):
    """Rows whose true margin ~ N(mu, 13^2); forecasts carry sd 13 * sd_scale and mean mu + bias (+ noise)."""
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1000)      # model noise drawn separately so outcomes match across models
    rows = []
    for d in range(n_days):
        day = f"2025-{1 + d // 28:02d}-{1 + d % 28:02d}"
        for k in range(per_day):
            mu = rng.normal(0, 6)
            m = mu + rng.normal(0, 13)
            fm = mu + bias + rng2.normal(0, noise_model) if noise_model else mu + bias
            v = (13 * sd_scale) ** 2
            rows.append({"game_id": f"g{d}_{k}", "game_date": day, "margin": m, "home_win": int(m > 0),
                         "margin_mean": fm, "margin_var": v, "p_home": float(E.Phi(fm / math.sqrt(v)))})
    return rows


class TestEvaluate(unittest.TestCase):
    def test_crps_closed_form_matches_numeric(self):
        y, mu, v = 3.0, 1.0, 4.0
        xs = np.linspace(-40, 40, 200001)
        F = E.Phi((xs - mu) / 2.0)
        num = np.trapezoid((F - (xs >= y)) ** 2, xs)
        self.assertAlmostEqual(float(E.NormalCrps(y, mu, v)), num, places=3)
        self.assertAlmostEqual(E.MixCrps(y, [1.0], [mu], [v]), float(E.NormalCrps(y, mu, v)), places=9)
        self.assertAlmostEqual(E.MixCrps(y, [0.5, 0.5], [mu, mu], [v, v]), float(E.NormalCrps(y, mu, v)), places=9)

    def test_phi_invphi(self):
        for p in (0.01, 0.2, 0.5, 0.77, 0.999):
            self.assertAlmostEqual(float(E.Phi(E.InvPhi(p))), p, places=6)

    def test_corp_decomposition_and_calibration(self):
        rows = FakeRows()
        p = np.array([r["p_home"] for r in rows])
        y = np.array([r["home_win"] for r in rows])
        c = E.Corp(p, y)
        self.assertAlmostEqual(np.mean((p - y) ** 2), c["mcb"] - c["dsc"] + c["unc"], places=10)
        self.assertLess(c["mcb"], 0.01)
        s = E.Score(rows)
        self.assertAlmostEqual(s["coverage80"], 0.8, delta=0.03)
        self.assertLess(s["pit_max_dev"], 0.03)
        self.assertAlmostEqual(s["spread_skill"], 1.0, delta=0.05)
        bad = E.Score(FakeRows(sd_scale=0.5))
        self.assertLess(bad["coverage80"], 0.6)
        self.assertGreater(bad["crps"], s["crps"])

    def test_paired_bootstrap_and_elo_rows(self):
        good, bad = FakeRows(seed=1), FakeRows(seed=1, bias=4.0)
        lg, lb = E.LossVectors(good), E.LossVectors(bad)
        b = E.PairedBootstrap(lg["log_loss"], lb["log_loss"], [r["game_date"] for r in good])
        self.assertLess(b["hi"], 0)
        er = E.EloRows([{"game_id": "x", "margin": 1, "home_win": 1}], {"x": 0.64})
        self.assertAlmostEqual(float(E.Phi(er[0]["margin_mean"] / 13.0)), 0.64, places=6)


class TestOverfit(unittest.TestCase):
    def test_pbo_noise_vs_signal(self):
        rng = np.random.default_rng(3)
        noise = rng.normal(0.65, 0.3, size=(12, 1600))
        self.assertTrue(0.25 < O.PboCscv(noise, blocks=10)["pbo"] < 0.75)
        sig = noise.copy()
        sig[0] -= 0.08
        self.assertLess(O.PboCscv(sig, blocks=10)["pbo"], 0.05)

    def test_selection_adjusted(self):
        rng = np.random.default_rng(4)
        days = np.repeat(np.arange(400), 4)
        base = rng.normal(0.65, 0.3, 1600)
        noise = base[None, :] + rng.normal(0, 0.05, size=(20, 1600))
        a = O.SelectionAdjusted(noise, base, days, reps=500)
        self.assertLess(a["adjusted_lo"], a["naive_lo"])
        self.assertLess(a["adjusted_lo"], 0)
        good = noise.copy()
        good[5] = base - 0.05 + rng.normal(0, 0.05, 1600)
        self.assertGreater(O.SelectionAdjusted(good, base, days, reps=500)["adjusted_lo"], 0)


class TestTrials(unittest.TestCase):
    def test_registry_counts_distinct(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.jsonl")
            T.Append("a", {"x": 1}, "2025-26", {"n": 5}, path=p)
            T.Append("a", {"x": 1}, "2025-26", {"n": 5}, path=p)
            T.Append("a", {"x": 2}, "2025-26", {"n": 5}, path=p)
            T.Append("b", {}, "fwd", {"n": 5}, path=p)
            self.assertEqual(T.Count("2025-26", path=p), 2)


def Perturbed(rows, cut):
    rng = np.random.default_rng(9)
    out = []
    for r in rows:
        if r["game_date"] >= cut:
            m = float(rng.normal(0, 30))
            r = dict(r, margin=m, home_win=int(m > 0))
        out.append(r)
    return out


class TestPost(unittest.TestCase):
    def test_emos_widens_underdispersed_and_no_lookahead(self):
        rows = FakeRows(sd_scale=0.6, bias=2.0)
        out = PP.Emos(rows, window=120, min_rows=200)
        late = [r for r in out if r["game_date"] >= "2025-06-01"]
        self.assertGreater(np.mean([r["margin_var"] for r in late]), 1.5 * (13 * 0.6) ** 2)
        self.assertLess(E.Score(late)["crps"], E.Score([r for r in rows if r["game_date"] >= "2025-06-01"])["crps"])
        cut = "2025-05-01"
        a = [r for r in out if r["game_date"] <= cut]
        b = [r for r in PP.Emos(Perturbed(rows, cut), window=120, min_rows=200) if r["game_date"] <= cut]
        self.assertEqual([(r["margin_mean"], r["margin_var"]) for r in a], [(r["margin_mean"], r["margin_var"]) for r in b])

    def test_temperature_no_lookahead(self):
        rows = FakeRows(sd_scale=0.6)
        cut = "2025-05-01"
        a = [r["p_home"] for r in PP.Temperature(rows) if r["game_date"] <= cut]
        b = [r["p_home"] for r in PP.Temperature(Perturbed(rows, cut)) if r["game_date"] <= cut]
        self.assertEqual(a, b)
        late = [r for r in PP.Temperature(rows) if r["game_date"] >= "2025-06-01"]
        self.assertLess(late[-1]["temperature"][1], 1.0)        # overconfident input -> shrink logits

    def test_context_stack_weights_and_no_lookahead(self):
        good, bad = FakeRows(seed=5), FakeRows(seed=5, noise_model=10.0)
        ctx = {r["game_id"]: ("early" if r["game_date"] < "2025-02-01" else "normal") for r in good}
        out, names, wg, wc = PP.ContextStack({"good": good, "bad": bad}, ctx, min_rows=200)
        self.assertGreater(wg[names.index("good")], 0.75)
        self.assertAlmostEqual(sum(out[-1]["mix"][0]), 1.0, places=9)
        cut = "2025-05-01"
        g2, b2 = Perturbed(good, cut), Perturbed(bad, cut)
        out2 = PP.ContextStack({"good": g2, "bad": b2}, ctx, min_rows=200)[0]
        A = [r["p_home"] for r in out if r["game_date"] <= cut]
        B = [r["p_home"] for r in out2 if r["game_date"] <= cut]
        self.assertEqual(A, B)


class TestAvailability(unittest.TestCase):
    def test_status_model_and_scenarios(self):
        sm = StatusModel()
        team = "AAA"
        for g in range(40):
            rows = [{"player_id": p, "team": team, "minutes": 30.0 if p < 5 else 15.0} for p in range(10)
                    if not (p == 3 and g % 2 == 1) and not (p == 9 and g > 30)]
            sm.Update(team, rows)
        probs = sm.Probs(team)
        self.assertGreater(probs[0], 0.85)
        self.assertLess(probs[9], probs[0])
        sc = sm.Scenarios(team)
        self.assertAlmostEqual(sum(w for w, _ in sc), 1.0, places=9)
        for w, m in sc:
            self.assertAlmostEqual(sum(m.values()), 240.0, delta=1.0)
        mean, var = Mixture([(0.5, 0.0, 1.0), (0.5, 2.0, 1.0)])
        self.assertAlmostEqual(mean, 1.0)
        self.assertAlmostEqual(var, 2.0)


class TestLearnedVariance(unittest.TestCase):
    def Fit(self, beta, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(1500, 6))
        y = X @ beta + rng.normal(0, 13, 1500)
        r = BayesLinear(6, 4.0, sigma2=169.0)
        for x, yy in zip(X, y):
            r.Add(x, yy)
        r.Fit()
        return LearnPriorScale(r, np.full(6, 4.0), iterations=30)

    def test_collapses_without_signal_and_recovers_with(self):
        self.assertLess(self.Fit(np.zeros(6)), 0.05)              # dropped family
        s = self.Fit(np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0]))  # true var 4 -> s ~ 1
        self.assertTrue(0.4 < s < 2.5, s)


class TestSkills(unittest.TestCase):
    def test_shrinkage_and_decay(self):
        sb = SkillBook()
        sb.Update([{"player_id": 1, "minutes": 36, "pts": 40, "fg3m": 5, "fg3a": 5}])
        m, _ = sb.Rate(1, "pts")
        self.assertTrue(16 < m < 40)                     # one big game is shrunk
        r3, _ = sb.Rate(1, "fg3")
        self.assertLess(r3, 0.4)
        for _ in range(200):
            sb.Update([{"player_id": 1, "minutes": 36, "pts": 10}])
        self.assertLess(sb.Rate(1, "pts")[0], 12)        # decay forgets the 40-point game


class TestPlayerKalman(unittest.TestCase):
    def test_recovers_skill_and_inflates_offseason(self):
        games, recs, truth = MakeLeague(n_days=60, games_per_day=3, seed=4)
        a = REGISTRY["player_kalman"]()
        Walk(a, games, records=lambda g: recs[g["game_id"]])
        ps = [p for p in truth["skill"] if p in a.m]
        team_eff = {p: truth["team_strength"][t] for t, pl in truth["players"].items() for p in pl}
        tgt = [truth["skill"][p] / 5 + team_eff[p] / 5 for p in ps]
        self.assertGreater(np.corrcoef([a.m[p] for p in ps], tgt)[0, 1], 0.3)
        p0 = ps[0]
        before = a.P[p0]
        a.BeforeDay("2025-10-20", 120)
        self.assertGreater(a.P[p0], before + 2.9)


if __name__ == "__main__":
    unittest.main()
