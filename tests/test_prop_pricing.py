import unittest

from Props.BuildPricingPanel import QuoteAt
from Props.ExportPropPricing import Decide, FEE


def Row(p, ask, bid, yes, close_ask=None, close_bid=None):
    q = {"t": 0, "ask": ask, "bid": bid}
    c = {"t": 1, "ask": close_ask, "bid": close_bid} if close_ask is not None else None
    return {"model_p": p, "yes": yes, "quotes": {"T60": q, "T2": c}}


class PropPricingTests(unittest.TestCase):
    def test_yes_edge_uses_ask_plus_fee(self):
        d = Decide(Row(0.60, 0.50, 0.45, 1), "T60", "yes", 0.03, 1.0)
        self.assertAlmostEqual(d["edge"], 0.60 - 0.50 - FEE)
        self.assertAlmostEqual(d["pnl"], 1 - 0.50 - FEE)

    def test_no_side_uses_one_minus_bid(self):
        d = Decide(Row(0.30, 0.50, 0.45, 1), "T60", "no", 0.03, 1.0)
        self.assertEqual(d["side"], "no")
        self.assertAlmostEqual(d["fill"], 0.55)
        self.assertAlmostEqual(d["pnl"], 0 - 0.55 - FEE)

    def test_cut_and_max_price(self):
        self.assertIsNone(Decide(Row(0.54, 0.50, 0.45, 1), "T60", "yes", 0.03, 1.0))
        self.assertIsNone(Decide(Row(0.95, 0.70, 0.65, 1), "T60", "yes", 0.03, 0.60))

    def test_quote_never_uses_future_candle_and_respects_staleness(self):
        candles = [{"t": 100, "ask": 0.4}, {"t": 200, "ask": 0.9}]
        self.assertEqual(QuoteAt(candles, 150)["t"], 100)
        self.assertIsNone(QuoteAt(candles, 99))
        self.assertIsNone(QuoteAt(candles, 200 + 3601))

    def test_clv_vs_close_mid(self):
        d = Decide(Row(0.60, 0.50, 0.45, 0, close_ask=0.56, close_bid=0.52), "T60", "yes", 0.03, 1.0)
        self.assertAlmostEqual(d["clv"], 0.54 - 0.50)


if __name__ == "__main__":
    unittest.main()