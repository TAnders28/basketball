"""Edge / CLV / fee-estimate helpers. No network."""

import unittest

from Backtest import Odds


class EdgeHelperTests(unittest.TestCase):
    def test_edge_uses_tim_fee_estimate(self):
        self.assertEqual(Odds.FEE_BASIS, "tim_estimate_0.02_per_contract")
        self.assertAlmostEqual(Odds.Edge(0.60, 0.55), 0.03)
        self.assertAlmostEqual(Odds.Edge(0.55, 0.55), -0.02)
        self.assertAlmostEqual(Odds.BreakevenAsk(0.55), 0.57)

    def test_devig_unchanged(self):
        h, a = Odds.FairProbs(-150, 130)
        self.assertAlmostEqual(h + a, 1.0, places=9)
        self.assertGreater(h, Odds.ImpliedProb(-150) - 0.05)

    def test_market_favorite_baseline(self):
        self.assertEqual(Odds.MarketFavorite(0.55, 0.45), "home")
        self.assertEqual(Odds.MarketFavorite(0.40, 0.60), "away")
        self.assertEqual(Odds.MarketFavorite(0.5, 0.5), "home")

    def test_clv_when_prices_exist(self):
        clv = Odds.ClosingLineValue(0.62, 0.58)
        self.assertTrue(clv["beats_close_after_fee"])
        self.assertEqual(clv["fee_basis"], Odds.FEE_BASIS)
        self.assertAlmostEqual(clv["edge_vs_close"], 0.02)

    def test_without_prices_no_pnl(self):
        rep = Odds.ReportWithoutPrices(7, 10)
        self.assertEqual(rep["accuracy"], 0.7)
        self.assertIsNone(rep["profit"])
        self.assertIsNone(rep["roi"])


if __name__ == "__main__":
    unittest.main()
