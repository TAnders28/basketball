"""Tests for T-60 quotes, implied NO asks, and leak-free count-model features."""
import math
import unittest

from Props.PullProps import QuoteAtT60
from Props.Model import (
    AttachCountPredictions, BuildFeatures, FairGe, FitCountParams, Shrink,
)
from tests.test_threes_props import G, P


class QuoteTests(unittest.TestCase):
    def test_t60_uses_last_candle_at_or_before_target(self):
        tip = 1_000_000
        candles = [
            {"t": tip - 7200, "bid": 0.40, "ask": 0.50},
            {"t": tip - 3600, "bid": 0.41, "ask": 0.45},
            {"t": tip - 1800, "bid": 0.60, "ask": 0.70},  # after T-60, must be ignored
        ]
        q = QuoteAtT60(candles, tip, stale_s=3600)
        self.assertEqual(q["t"], tip - 3600)
        self.assertAlmostEqual(q["ask"], 0.45)
        self.assertAlmostEqual(q["bid"], 0.41)
        self.assertAlmostEqual(q["mid"], 0.43)
        self.assertAlmostEqual(q["spread"], 0.04)
        self.assertEqual(q["age_s"], 0)

    def test_t60_rejects_stale_and_missing(self):
        tip = 1_000_000
        candles = [{"t": tip - 3600 - 3601, "bid": 0.2, "ask": 0.3}]
        self.assertIsNone(QuoteAtT60(candles, tip, stale_s=3600))
        self.assertIsNone(QuoteAtT60([], tip))
        # Only post-T-60 candles
        self.assertIsNone(QuoteAtT60([{"t": tip - 100, "bid": 0.2, "ask": 0.3}], tip))

    def test_implied_no_ask_and_edge(self):
        p, fee, cut = 0.40, 0.02, 0.03
        yes_ask, yes_bid = 0.50, 0.46
        no_ask = 1.0 - yes_bid  # 0.54, implied
        e_yes = p - (yes_ask + fee)          # 0.40 - 0.52 = -0.12
        e_no = (1.0 - p) - (no_ask + fee)    # 0.60 - 0.56 = 0.04
        self.assertLess(e_yes, cut)
        self.assertGreaterEqual(e_no, cut)
        # One side only: NO
        side = "no" if e_no > e_yes and e_no >= cut else "yes"
        self.assertEqual(side, "no")
        pnl_if_no_hits = 1 - (no_ask + fee)
        self.assertAlmostEqual(pnl_if_no_hits, 0.44)


class CountModelTests(unittest.TestCase):
    def test_pts_features_exclude_same_game(self):
        players = [
            P(1, "A", "GSW", "2024-11-01", "0022400001", 30, 3, 7, "GSW vs. LAL"),
            P(1, "A", "GSW", "2024-11-03", "0022400002", 28, 1, 4, "GSW @ OKC"),
        ]
        players[0]["PTS"] = 20
        players[0]["REB"] = 5
        players[0]["AST"] = 4
        players[1]["PTS"] = 8
        players[1]["REB"] = 1
        players[1]["AST"] = 0
        games = [
            G("0022400001", "2024-11-01", "GSW", "LAL"),
            G("0022400002", "2024-11-03", "OKC", "GSW"),
        ]
        rows, pst, _ = BuildFeatures(players, games)
        r1 = next(r for r in rows if r["game_id"] == "0022400001")
        r2 = next(r for r in rows if r["game_id"] == "0022400002")
        self.assertEqual(r1["prior_pts"], 0)
        self.assertEqual(r1["actual_pts"], 20)
        self.assertEqual(r2["prior_pts"], 20)
        self.assertEqual(r2["prior_reb"], 5)
        self.assertEqual(r2["prior_ast"], 4)
        self.assertEqual(r2["actual_pts"], 8)
        self.assertEqual(pst[1]["pts"], 28)

    def test_count_params_ignore_holdout_actuals(self):
        rows_p, games = [], []
        for i, d in enumerate(["2024-11-01", "2024-11-03", "2024-11-05", "2024-11-08",
                               "2024-11-10", "2024-11-12", "2024-11-14", "2024-11-16"]):
            gid = f"0022401{i:03d}"
            rec = P(1, "A", "GSW", d, gid, 30, 2, 6, "GSW vs. LAL", "22024")
            rec["PTS"] = 15
            rec["REB"] = 4
            rec["AST"] = 3
            rows_p.append(rec)
            games.append(G(gid, d, "GSW", "LAL"))
        rec = P(1, "A", "GSW", "2025-11-01", "0022501000", 40, 8, 12, "GSW vs. LAL", "22025")
        rec["PTS"] = 50
        rec["REB"] = 20
        rec["AST"] = 15
        rows_p.append(rec)
        games.append(G("0022501000", "2025-11-01", "GSW", "LAL"))
        feats, _, _ = BuildFeatures(rows_p, games)
        train = [r for r in feats if r["season"] == 2024]
        hold = [r for r in feats if r["season"] == 2025]
        self.assertEqual(hold[0]["prior_pts"], 15 * 8)
        params = FitCountParams(train, "pts")
        AttachCountPredictions(hold, params, "pts")
        self.assertIn("lam_pts", hold[0])
        self.assertGreater(hold[0]["lam_pts"], 0)
        # Holdout actual 50 must not be in the prior used for lambda.
        self.assertEqual(hold[0]["prior_pts"], 120)
        p = FairGe(hold[0], "pts", 20, params)
        self.assertGreater(p, 0)
        self.assertLess(p, 1)


if __name__ == "__main__":
    unittest.main()
