"""Unit tests for the V2 Bayesian toolkit and diagnostics (synthetic data only, single process, < few seconds)."""

import math
import unittest

import numpy as np

from Matchups.V2 import Bayes as B
from Matchups.V2 import Diagnostics as G


class TestConjugate(unittest.TestCase):
    def test_gamma_poisson_recovers_rate_and_shrinks(self):
        rng = np.random.default_rng(0)
        g = B.GammaPoisson(1.0, 50.0)
        g.Add(rng.poisson(1.3 * 5000), 5000)
        self.assertAlmostEqual(g.Mean(), 1.3, delta=0.03)
        small = B.GammaPoisson(1.0, 50.0)
        small.Add(20, 10)                      # raw 2.0, heavily shrunk toward 1.0
        self.assertLess(small.Mean(), 1.2)
        m, v = small.PredictiveMoments(10)
        self.assertGreater(v, m)               # negative binomial predictive is over-dispersed

    def test_beta_binomial_and_decay(self):
        b = B.BetaBinomial(0.5, 20)
        b.Add(80, 100)
        self.assertAlmostEqual(b.Mean(), (10 + 80) / 120, places=9)
        before = b.Var()
        b.Decay(0.5)
        self.assertGreater(b.Var(), before)    # forgetting widens the posterior
        lo, hi = b.Interval()
        self.assertTrue(lo < b.Mean() < hi)

    def test_empirical_bayes_beta(self):
        rng = np.random.default_rng(1)
        p = rng.beta(0.35 * 60, 0.65 * 60, size=400)
        n = rng.integers(50, 400, size=400)
        k = rng.binomial(n, p)
        m, s = B.EmpiricalBayesBeta(k, n)
        self.assertAlmostEqual(m, 0.35, delta=0.01)
        self.assertTrue(25 < s < 140, s)

    def test_empirical_bayes_normal(self):
        rng = np.random.default_rng(2)
        theta = rng.normal(1.0, 2.0, 500)
        se2 = np.full(500, 4.0)
        x = theta + rng.normal(0, 2.0, 500)
        mu, tau2, post, pv = B.EmpiricalBayesNormal(x, se2)
        self.assertAlmostEqual(mu, 1.0, delta=0.3)
        self.assertAlmostEqual(tau2, 4.0, delta=1.2)
        self.assertLess(np.mean((post - theta) ** 2), np.mean((x - theta) ** 2))   # shrinkage helps


class TestBayesLinear(unittest.TestCase):
    def test_matches_closed_form_and_recovers(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 4))
        beta = np.array([1.0, -2.0, 0.5, 0.0])
        y = X @ beta + rng.normal(0, 1.0, 300)
        r = B.BayesLinear(4, 10.0, sigma2=1.0)
        for x, yy in zip(X, y):
            r.Add(x, yy)
        r.Fit()
        P = X.T @ X + np.eye(4) / 10.0
        np.testing.assert_allclose(r.mean, np.linalg.solve(P, X.T @ y), atol=1e-8)
        np.testing.assert_allclose(r.cov, np.linalg.inv(P), atol=1e-10)
        z = (r.mean - beta) / np.sqrt(np.diag(r.cov))
        self.assertTrue(np.all(np.abs(z) < 4))

    def test_prior_dominates_with_no_data_and_sensitivity(self):
        r = B.BayesLinear(2, 1.0, prior_mean=[3.0, -1.0], sigma2=1.0)
        r.Add([1.0, 0.0], 0.0)
        r.Fit()
        self.assertTrue(0 < r.mean[0] < 3)
        self.assertEqual(r.mean[1], -1.0)

        def Run(scale):
            q = B.BayesLinear(1, 1.0 * scale, sigma2=1.0)
            for _ in range(3):
                q.Add([1.0], 2.0)
            q.Fit()
            return [q.mean[0]]
        s = G.PriorSensitivity(Run)
        self.assertGreater(s[2.0]["max_abs"], 0)       # a little data: prior matters
        self.assertLess(s[2.0]["max_abs"], 1.0)


class TestKalmanCalibration(unittest.TestCase):
    def test_kalman_tracks_and_is_calibrated(self):
        """Posterior predictive check: data generated from the model -> ~80% coverage, flat PIT, low ECE."""
        from Matchups.V2.Approaches.Common import KalmanCore
        rng = np.random.default_rng(4)
        core = KalmanCore()
        teams = core.TEAMS
        off = rng.normal(0, 3, 30)
        dfn = rng.normal(0, 3, 30)
        cov = np.array([[110.0, 25.5], [25.5, 110.0]])      # per-team var 110, margin var 169
        means, vars_, actual = [], [], []
        for day in range(150):
            core.k.Predict(1)
            off += rng.normal(0, math.sqrt(0.02), 30)
            dfn += rng.normal(0, math.sqrt(0.02), 30)
            order = rng.permutation(30)
            for a, b in zip(order[:15], order[15:]):
                h, w = teams[a], teams[b]
                m, v = core.Forecast(h, w)
                e = rng.multivariate_normal([0, 0], cov)
                hs = 112 + 1.25 + off[a] + dfn[b] + e[0]
                as_ = 112 - 1.25 + off[b] + dfn[a] + e[1]
                if day >= 40:
                    means.append(m)
                    vars_.append(v)
                    actual.append(hs - as_)
                core.Update({"home_team": h, "away_team": w, "home_score": hs, "away_score": as_})
        cov80 = G.Coverage(means, vars_, actual, 0.8)
        self.assertAlmostEqual(cov80, 0.8, delta=0.04)
        pit = G.Pit(means, vars_, actual)
        self.assertLess(max(abs(x - 0.1) for x in pit), 0.03)
        p = [B.PredictiveWinProb(m, v) for m, v in zip(means, vars_)]
        ece, _ = G.Ece(p, [int(a > 0) for a in actual])
        self.assertLess(ece, 0.04)
        est = core.k.x[:30] - (off - off.mean())
        self.assertLess(np.corrcoef(core.k.x[:30], off)[0, 1], 1.0)
        self.assertGreater(np.corrcoef(core.k.x[:30], off)[0, 1], 0.7)


class TestLaplace(unittest.TestCase):
    def test_laplace_logistic_vs_metropolis(self):
        rng = np.random.default_rng(5)
        X = np.c_[np.ones(200), rng.normal(size=200)]
        y = (rng.random(200) < 1 / (1 + np.exp(-(0.3 + 1.0 * X[:, 1])))).astype(float)
        w, cov = B.LaplaceLogistic(X, y, [0, 0], 4.0)

        def Lp(b):
            z = X @ b
            return float(np.sum(y * z - np.logaddexp(0, z)) - np.sum(b ** 2) / 8)
        draws, acc = B.Metropolis(Lp, w, steps=6000, scale=0.15, seed=1)
        draws = draws[1000:]
        self.assertTrue(0.15 < acc < 0.8)
        np.testing.assert_allclose(draws.mean(0), w, atol=0.06)
        np.testing.assert_allclose(draws.std(0), np.sqrt(np.diag(cov)), rtol=0.25)

    def test_laplace_poisson_recovers(self):
        rng = np.random.default_rng(6)
        X = np.c_[np.ones(500), rng.normal(size=500)]
        e = rng.uniform(5, 20, 500)
        y = rng.poisson(e * np.exp(-0.2 + 0.4 * X[:, 1]))
        b, cov = B.LaplacePoisson(X, y, e, [0, 0], 1.0)
        self.assertTrue(np.all(np.abs((b - [-0.2, 0.4]) / np.sqrt(np.diag(cov))) < 4))

    def test_predictive_win_prob(self):
        self.assertAlmostEqual(B.PredictiveWinProb(0.0, 169.0), 0.5)
        self.assertGreater(B.PredictiveWinProb(5.0, 169.0), B.PredictiveWinProb(5.0, 400.0))   # uncertainty pulls to 0.5


class TestDiagnostics(unittest.TestCase):
    def test_ece_logloss_coverage(self):
        p = np.full(1000, 0.7)
        y = (np.random.default_rng(7).random(1000) < 0.7).astype(int)
        self.assertLess(G.Ece(p, y)[0], 0.04)
        self.assertAlmostEqual(G.LogLoss([0.5], [1]), math.log(2))
        self.assertEqual(G.Coverage([0, 0], [1, 1], [0.1, 5.0], 0.8), 0.5)


if __name__ == "__main__":
    unittest.main()
