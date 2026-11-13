"""Unit tests for residual-on-market offset and leak-free feature logic."""
import math
import unittest

import numpy as np

from Backtest.ResidualSearch import (
    AttachResiduals, BuildGrid, FitLogisticOffset, FitPredictLgbm, FitPredictXgb,
    LineMoveFromCandles, MonthFolds, PredictLogisticOffset, ResidualBook,
    ScheduleBook, BURNIN_END, ClipP, Logit, Sigmoid,
)


class OffsetTests(unittest.TestCase):
    def test_logistic_zero_features_equals_market(self):
        rng = np.random.default_rng(0)
        p_mkt = rng.uniform(0.2, 0.8, size=400)
        y = (rng.random(400) < p_mkt).astype(float)
        offset = np.array([Logit(p) for p in p_mkt])
        X = np.zeros((400, 3))
        beta = FitLogisticOffset(X, y, offset, l2=1.0, fit_intercept=True)
        pred = PredictLogisticOffset(beta, X, offset, fit_intercept=True)
        # Intercept may absorb a tiny calibration; predictions should stay near the market.
        np.testing.assert_allclose(pred, p_mkt, atol=0.03)
        # The feature weights themselves must stay at 0 (no residual to learn).
        np.testing.assert_allclose(beta[1:], 0.0, atol=1e-6)

    def test_logistic_offset_is_frozen(self):
        # A useful feature should move predictions off the market in the right direction,
        # but the market logit is added, not estimated.
        p_mkt = np.full(200, 0.5)
        feat = np.r_[np.ones(100), np.zeros(100)]
        p_true = np.where(feat == 1, 0.8, 0.3)
        rng = np.random.default_rng(1)
        y = (rng.random(200) < p_true).astype(float)
        offset = np.array([Logit(p) for p in p_mkt])
        X = feat.reshape(-1, 1)
        beta = FitLogisticOffset(X, y, offset, l2=0.1, fit_intercept=True)
        pred = PredictLogisticOffset(beta, X, offset, fit_intercept=True)
        self.assertGreater(pred[:100].mean(), pred[100:].mean() + 0.2)

    def test_lgbm_raw_score_adds_offset(self):
        p_mkt = np.linspace(0.2, 0.8, 80)
        y = (p_mkt >= 0.5).astype(float)
        offset = np.array([Logit(p) for p in p_mkt])
        X = np.zeros((80, 2))
        params = dict(num_leaves=8, min_child_samples=20, learning_rate=0.1,
                      n_estimators=20, max_depth=2, reg_lambda=1.0)
        pred = FitPredictLgbm(X, y, offset, X, offset, params)
        # With no features the model can only learn a constant residual; market shape remains.
        self.assertTrue(np.all(np.diff(pred) >= -1e-12))
        self.assertGreater(pred[-1], pred[0])
        corr = np.corrcoef(pred, p_mkt)[0, 1]
        self.assertGreater(corr, 0.95)

    def test_xgb_base_margin_adds_offset(self):
        p_mkt = np.linspace(0.2, 0.8, 80)
        y = (p_mkt >= 0.5).astype(float)
        offset = np.array([Logit(p) for p in p_mkt])
        X = np.zeros((80, 2))
        params = dict(max_depth=2, min_child_weight=1, learning_rate=0.1,
                      n_estimators=20, reg_lambda=1.0, subsample=1.0)
        pred = FitPredictXgb(X, y, offset, X, offset, params)
        self.assertGreater(pred[-1], pred[0])
        corr = np.corrcoef(pred, p_mkt)[0, 1]
        self.assertGreater(corr, 0.95)


class LeakageTests(unittest.TestCase):
    def test_residual_book_same_day_does_not_leak(self):
        book = ResidualBook(prior_n=10)
        # Before any games, both views are 0.
        self.assertEqual(book.View("AAA", "BBB"), (0.0, 0.0))
        day1 = [
            {"home_team": "AAA", "away_team": "BBB", "home_win": 1, "p_mkt": 0.4},
            {"home_team": "AAA", "away_team": "CCC", "home_win": 0, "p_mkt": 0.6},
        ]
        v0 = book.View("AAA", "CCC")
        # Same-day second game must still see the opening book.
        self.assertEqual(v0, (0.0, 0.0))
        book.ApplyDay(day1)
        rh, ra = book.View("AAA", "DDD")
        # AAA: (1-0.4) + (0-0.6) = 0.0 over 2 games, shrunk by prior 10 -> 0
        self.assertAlmostEqual(rh, 0.0)
        book.ApplyDay([{"home_team": "AAA", "away_team": "DDD", "home_win": 1, "p_mkt": 0.5}])
        rh2, _ = book.View("AAA", "EEE")
        self.assertGreater(rh2, rh)

    def test_attach_residuals_groups_by_day(self):
        games = [
            {"game_id": "a", "game_date": "2025-10-21", "home_team": "AAA", "away_team": "BBB",
             "home_win": 1, "p_mkt": 0.3},
            {"game_id": "b", "game_date": "2025-10-21", "home_team": "CCC", "away_team": "DDD",
             "home_win": 0, "p_mkt": 0.7},
            {"game_id": "c", "game_date": "2025-10-22", "home_team": "AAA", "away_team": "CCC",
             "home_win": 1, "p_mkt": 0.5},
        ]
        AttachResiduals(games)
        self.assertEqual(games[0]["resid_home"], games[1]["resid_home"])  # both 0 on day 1
        self.assertEqual(games[0]["resid_home"], 0.0)
        self.assertNotAlmostEqual(games[2]["resid_home"], 0.0)  # AAA won as a 0.3 dog
        self.assertGreater(games[2]["resid_home"], 0.0)

    def test_month_folds_train_only_on_earlier_dates(self):
        games = []
        days = (["2025-10-21", "2025-11-10", "2025-12-01"]
                + [f"2025-12-{d:02d}" for d in range(2, 10)]
                + [f"2026-01-{d:02d}" for d in range(1, 5)])
        for i, d in enumerate(days):
            games.append({"game_date": d, "month": d[:7], "game_id": str(i),
                          "scored": d >= BURNIN_END})
        folds = MonthFolds(games)
        self.assertTrue(folds)
        for f in folds:
            train_dates = [games[i]["game_date"] for i in f["train_idx"]]
            test_dates = [games[i]["game_date"] for i in f["test_idx"]]
            self.assertTrue(test_dates)
            self.assertLess(max(train_dates), min(test_dates))
            self.assertTrue(all(d >= BURNIN_END for d in test_dates))
            self.assertTrue(all(d.startswith(f["month"]) for d in test_dates))
        # Burn-in 2025-12-01 is training for the December fold, not test.
        dec = next(f for f in folds if f["month"] == "2025-12")
        self.assertIn("2025-12-01", [games[i]["game_date"] for i in dec["train_idx"]])
        self.assertNotIn("2025-12-01", [games[i]["game_date"] for i in dec["test_idx"]])

    def test_line_move_ignores_candles_after_t60(self):
        sched_ts = 1_000_000
        t60 = sched_ts - 3600
        candles = [
            {"t": t60 - 7200, "bid": 0.40, "ask": 0.42},
            {"t": t60 - 60, "bid": 0.50, "ask": 0.52},
            {"t": t60 + 1, "bid": 0.90, "ask": 0.92},  # after T-60, must be ignored
            {"t": sched_ts, "bid": 0.99, "ask": 1.00},
        ]
        info = LineMoveFromCandles(candles, sched_ts)
        self.assertAlmostEqual(info["mid_open"], 0.41)
        self.assertAlmostEqual(info["mid_t60"], 0.51)
        self.assertAlmostEqual(info["line_move"], 0.10)
        self.assertEqual(info["n_candles"], 2)

    def test_travel_and_rest_use_prior_game_only(self):
        book = ScheduleBook()
        v1 = book.View("2025-10-21", "NYK", "BOS", "NYK")
        self.assertIsNone(v1["home_rest"])
        self.assertEqual(v1["rest_missing"], 1.0)
        book.Apply("2025-10-21", "NYK", "BOS", "NYK")
        # NYK at home again two days later: 1 day of rest, no travel. BOS travels to CHA.
        v2 = book.View("2025-10-23", "CHA", "BOS", "CHA")
        self.assertEqual(v2["away_rest"], 1)
        self.assertEqual(v2["away_b2b"], 0.0)
        self.assertGreater(v2["away_travel"], 0.0)
        # Back-to-back: next night
        book.Apply("2025-10-23", "CHA", "BOS", "CHA")
        v3 = book.View("2025-10-24", "BOS", "CHA", "BOS")
        self.assertEqual(v3["home_b2b"], 1.0)
        self.assertEqual(v3["away_b2b"], 1.0)
        self.assertEqual(v3["home_rest"], 0)
        # A later same-day view before Apply must not see the later game; already tested by order.

    def test_grid_at_least_2000(self):
        g = BuildGrid()
        self.assertGreaterEqual(len(g), 2000)
        self.assertEqual(len({c["config_id"] for c in g}), len(g))
        self.assertTrue({"logistic", "lgbm", "xgb"} <= {c["family"] for c in g})


if __name__ == "__main__":
    unittest.main()
