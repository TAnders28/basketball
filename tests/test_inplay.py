import unittest

from InPlay.Align import AsOf, Elapsed, EspnStates, IsoTs, KalshiMid, MonotoneWall, SeasonOf, Team
from InPlay.Engine import Game, RunGame, TradePnl
from InPlay.Execution import Fee, TakerBuy, TakerSell
from InPlay.Load import SplitOf
from InPlay.PullKalshi import NormalizeCandle, ParseEventTicker
from InPlay.PullPolymarket import MarketRow as PmRow
from InPlay.Stats import MaxDrawdown, Pbo, RatioBootstrap, Traversals


def MakeGame(mids, won=1, pre=0.6, el=None, sd=None, spread=0.01, per=None):
    n = len(mids)
    el = el or [60.0 * k for k in range(n)]
    return Game(gid="g", day="2026-01-01", season=2026, stype=2, venue="kalshi", pre=pre, won=won,
                t=[60 * k for k in range(n)], el=el, per=per or [min(4, int(e // 720) + 1) for e in el],
                sd=sd or [0] * n, wp=[None] * n, bid=[None if m is None else m - spread / 2 for m in mids],
                ask=[None if m is None else m + spread / 2 for m in mids], mid=mids, trlo=[None] * n, trhi=[None] * n,
                pm=[None] * n, pmage=[None] * n, total=None, spread=None)


class TestParsers(unittest.TestCase):
    def test_event_ticker(self):
        self.assertEqual(ParseEventTicker("KXNBAGAME-26NOV11BOSTOR", ticker="KXNBAGAME-26NOV11BOSTOR-TOR"),
                         {"date": "2026-11-11", "away": "BOS", "home": "TOR"})
        self.assertEqual(ParseEventTicker("KXNBAGAME-25APR15ATLORL", ticker="KXNBAGAME-25APR15ATLORL-ATL"),
                         {"date": "2025-04-15", "away": "ATL", "home": "ORL"})
        self.assertIsNone(ParseEventTicker("KXNBA-25"))

    def test_candle_both_schemas(self):
        old = {"end_period_ts": 10, "price": {"close": "0.76", "low": "0.74", "high": "0.77"}, "volume": "5",
               "yes_ask": {"close": "0.76"}, "yes_bid": {"close": "0.75"}, "open_interest": "1"}
        new = {"end_period_ts": 10, "price": {"close_dollars": "0.76", "low_dollars": "0.74", "high_dollars": "0.77"},
               "volume_fp": "5", "yes_ask": {"close_dollars": "0.76"}, "yes_bid": {"close_dollars": "0.75"}, "open_interest_fp": "1"}
        self.assertEqual(NormalizeCandle(old), NormalizeCandle(new))
        self.assertEqual(NormalizeCandle(new)["bid"], 0.75)

    def test_pm_row_home_token(self):
        raw = {"slug": "nba-bos-tor-2026-11-11", "sportsMarketType": "moneyline", "outcomes": '["Celtics", "Raptors"]',
               "clobTokenIds": '["a", "b"]', "outcomePrices": '["1", "0"]', "gameStartTime": "2026-11-12 00:00:00+00"}
        row = PmRow(raw)
        self.assertEqual((row["home"], row["away"], row["home_token"], row["home_won"]), ("TOR", "BOS", "b", 0))
        self.assertIsNone(PmRow({**raw, "sportsMarketType": "spreads"}))


class TestAlign(unittest.TestCase):
    def test_clock(self):
        self.assertEqual(Elapsed(1, "12:00"), 0)
        self.assertEqual(Elapsed(2, "0.0"), 1440)
        self.assertEqual(Elapsed(5, "5:00"), 2880)
        self.assertAlmostEqual(Elapsed(4, "24.5"), 2880 - 24.5)

    def test_monotone_wall_and_asof(self):
        plays = [{"seq": 2, "wall": "2026-01-01T00:01:00Z"}, {"seq": 1, "wall": "2026-01-01T00:00:00Z"},
                 {"seq": 3, "wall": "2025-12-31T23:59:00Z"}]
        ts = [t for t, _ in MonotoneWall(plays)]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(AsOf([1, 5, 9], ["a", "b", "c"], 4), "a")
        self.assertEqual(AsOf([1, 5, 9], ["a", "b", "c"], 5), "b")
        self.assertIsNone(AsOf([1, 5, 9], ["a", "b", "c"], 0))

    def test_espn_states_ffill_wp(self):
        game = {"plays": [{"id": "1", "seq": 1, "period": 1, "clock": "12:00", "hs": 0, "as": 0, "wall": "2026-01-01T00:00:00Z"},
                          {"id": "2", "seq": 2, "period": 1, "clock": "11:30", "hs": 2, "as": 0, "wall": "2026-01-01T00:00:30Z"}],
                "wp": [{"play": "1", "h": 0.6}]}
        s = EspnStates(game)
        self.assertEqual([r["wp"] for r in s], [0.6, 0.6])
        self.assertEqual(s[1]["el"], 30)

    def test_scoring_run_and_timeout(self):
        mk = lambda seq, hs, a, typ="Jump Shot": {"id": str(seq), "seq": seq, "period": 1, "clock": f"11:{59-seq:02d}",
                                                  "hs": hs, "as": a, "wall": f"2026-01-01T00:00:{seq:02d}Z", "type": typ}
        g = {"plays": [mk(1, 2, 0), mk(2, 4, 0), mk(3, 4, 3), mk(4, 4, 3, "Full Timeout"), mk(5, 4, 3, "Official TV Timeout")], "wp": []}
        s = EspnStates(g)
        self.assertEqual([r["run"] for r in s], [2, 4, -3, -3, -3])
        self.assertEqual(s[-1]["last_to"], s[3]["t"])

    def test_misc(self):
        self.assertEqual(Team("UTAH"), "UTA")
        self.assertEqual(Team("GS"), "GSW")
        self.assertIsNone(KalshiMid(0.0, 0.02))
        self.assertAlmostEqual(KalshiMid(0.40, 0.42), 0.41)
        self.assertEqual(SeasonOf(IsoTs("2025-10-25T00:00:00Z")), 2026)
        self.assertEqual(SeasonOf(IsoTs("2025-06-10T00:00:00Z")), 2025)
        self.assertEqual(SplitOf(2025), "tune")
        self.assertEqual(SplitOf(2026), "holdout")
        self.assertEqual(SplitOf(2027), "forward")


class TestFillsFees(unittest.TestCase):
    def test_fees(self):
        self.assertAlmostEqual(Fee(0.5, "kalshi_taker"), 0.0175)
        self.assertAlmostEqual(Fee(0.3, "kalshi_taker"), 0.0147)
        self.assertAlmostEqual(Fee(0.5, "kalshi_maker"), 0.004375)
        self.assertEqual(Fee(0.9, "tim_est_0.02"), 0.02)

    def test_no_side_quotes(self):
        # holding NO: buy at 1 - yes_bid, sell at 1 - yes_ask
        self.assertAlmostEqual(TakerBuy(0.60, 0.62, -1, 0.0), 0.40)
        self.assertAlmostEqual(TakerSell(0.60, 0.62, -1, 0.0), 0.38)
        self.assertAlmostEqual(TakerBuy(0.60, 0.62, 1, 0.005), 0.625)

    def test_settle_no_exit_fee(self):
        tr = {"entry": 0.40, "exit": 1.0, "settled": True, "maker": False}
        self.assertAlmostEqual(TradePnl(tr, "tim_est_0.02"), 0.58)
        tr2 = {"entry": 0.40, "exit": 0.60, "settled": False, "maker": False}
        self.assertAlmostEqual(TradePnl(tr2, "tim_est_0.02"), 0.16)
        self.assertAlmostEqual(TradePnl({**tr2, "maker": True}, "kalshi_taker"), 0.20 - 0.0175 * 0.24 - 0.07 * 0.24)


class TestEngine(unittest.TestCase):
    P = dict(pre_lo=0.55, pre_hi=0.65, entry=0.40, tmax=1440, tgt=0.60)

    def test_one_minute_latency_fill(self):
        g = MakeGame([0.6, 0.5, 0.39, 0.30, 0.45, 0.61, 0.65, 0.70])
        tr = RunGame("dip_fav", self.P, g, slip=0.0)
        self.assertEqual(len(tr), 1)
        self.assertAlmostEqual(tr[0]["entry"], 0.305)  # signal at 0.39, filled at next bar's ask
        self.assertAlmostEqual(tr[0]["exit"], 0.645)   # target seen at 0.61, sold next bar at bid
        self.assertEqual(tr[0]["why"], "target")

    def test_settle_when_no_exit(self):
        g = MakeGame([0.6, 0.39, 0.35, 0.3, 0.2], won=0)
        tr = RunGame("dip_fav", self.P, g, slip=0.0)
        self.assertEqual(tr[0]["exit"], 0.0)
        self.assertTrue(tr[0]["settled"])

    def test_away_favorite(self):
        g = MakeGame([0.4, 0.5, 0.61, 0.65, 0.4, 0.3], won=0, pre=0.4)  # away is the 60% favorite
        tr = RunGame("dip_fav", self.P, g, slip=0.0)
        self.assertEqual(tr[0]["side"], -1)
        self.assertAlmostEqual(tr[0]["entry"], 1 - 0.645)

    def test_no_lookahead(self):
        """Changing bars after the entry fill must not change the entry; changing the future never creates past trades."""
        a = MakeGame([0.6, 0.39, 0.40, 0.41, 0.42, 0.43])
        b = MakeGame([0.6, 0.39, 0.40, 0.90, 0.95, 0.99])
        ta, tb = RunGame("dip_fav", self.P, a, slip=0), RunGame("dip_fav", self.P, b, slip=0)
        self.assertEqual(ta[0]["entry"], tb[0]["entry"])
        c = MakeGame([0.6, 0.55, 0.50, 0.45, 0.42, 0.41])
        d = MakeGame([0.6, 0.55, 0.50, 0.45, 0.42, 0.10])
        self.assertEqual(RunGame("dip_fav", self.P, c, slip=0), [])
        tr = RunGame("dip_fav", self.P, d, slip=0)
        self.assertEqual(tr, [])  # signal on the last bar cannot fill (no next bar)

    def test_comeback_trailer(self):
        P = dict(close=0.05, tmax=2520, mode="pts", pts_lo=10, pts_hi=18, tgt=0.40)
        g = MakeGame([0.5, 0.6, 0.8, 0.8, 0.55, 0.5], pre=0.52, sd=[0, 5, 12, 12, 3, 0], won=0)
        tr = RunGame("comeback", P, g, slip=0.0)
        self.assertEqual(tr[0]["side"], -1)
        self.assertAlmostEqual(tr[0]["entry"], 1 - 0.795)
        self.assertEqual(RunGame("comeback", P, MakeGame([0.5, 0.6, 0.8, 0.8, 0.55, 0.5], pre=0.70,
                                                          sd=[0, 5, 12, 12, 3, 0]), slip=0.0), [])

    def test_xvenue_persistence_and_staleness(self):
        P = dict(gap=0.03, hold=2, close_gap=True, tmax=2820, max_tr=1, persist=2)
        g = MakeGame([0.50, 0.50, 0.50, 0.50, 0.50, 0.50])
        g.pm, g.pmage = [0.50, 0.56, 0.56, 0.56, 0.56, 0.56], [0, 0, 0, 0, 0, 0]
        tr = RunGame("xvenue", P, g, slip=0.0)
        self.assertEqual(tr[0]["side"], 1)
        g.pmage = [0, 0, 300, 300, 300, 300]  # stale Polymarket prints never trigger
        self.assertEqual(RunGame("xvenue", P, g, slip=0.0), [])
        g.pmage = [0] * 6
        g.pm = [0.50, 0.56, 0.50, 0.56, 0.50, 0.50]  # one-bar blips are not persistent
        self.assertEqual(RunGame("xvenue", P, g, slip=0.0), [])

    def test_maker_requires_trade_through(self):
        g = MakeGame([0.6, 0.39, 0.38, 0.37, 0.36, 0.6, 0.6])
        g.trlo = [None, None, 0.385, 0.36, None, None, None]
        tr = RunGame("dip_fav", self.P, g, slip=0.0, exec_mode="maker")
        self.assertTrue(tr[0]["maker"])
        self.assertAlmostEqual(tr[0]["entry"], 0.385)  # rested at the bid seen at signal (0.39-0.005)


class TestMakerExit(unittest.TestCase):
    def test_resting_target_needs_trade_through(self):
        P = dict(pre_lo=0.55, pre_hi=0.65, entry=0.40, tmax=1440, tgt=0.50)
        g = MakeGame([0.6, 0.39, 0.38, 0.45, 0.49, 0.49, 0.49])
        g.trlo = [None, None, 0.375, None, None, None, None]
        g.trhi = [None, None, None, 0.46, 0.50, 0.51, None]
        tr = RunGame("dip_fav", P, g, slip=0.0, exec_mode="maker2")
        self.assertEqual(tr[0]["why"], "maker_target")
        self.assertAlmostEqual(tr[0]["exit"], 0.50)
        self.assertAlmostEqual(TradePnl(tr[0], "kalshi_taker"), 0.50 - 0.385 - 0.0175 * 0.385 * 0.615 - 0.0175 * 0.25)


class TestStats(unittest.TestCase):
    def test_traversals(self):
        self.assertEqual(Traversals([0.5, 0.25, 0.5, 0.75, 0.5, 0.2], 0.3, 0.7)[0], 2)

    def test_drawdown(self):
        self.assertAlmostEqual(MaxDrawdown([1, -2, 1, -1]), 2)

    def test_ratio_bootstrap(self):
        p, lo, hi = RatioBootstrap([1, 1, 1], [1, 1, 1])
        self.assertEqual((p, lo, hi), (1.0, 1.0, 1.0))

    def test_pbo_bounds(self):
        import numpy as np
        rng = np.random.default_rng(0)
        val, n = Pbo(rng.normal(size=(64, 20)))
        self.assertTrue(0.0 <= val <= 1.0 and n > 0)

class TestRoundTwoThree(unittest.TestCase):
    def test_mm_fill_needs_trade_through_and_pull(self):
        from InPlay.Round3 import MmFills
        g = MakeGame([0.50, 0.50, 0.50, 0.50, 0.50, 0.50], won=1)
        g.wp = [0.5] * 6
        g.trlo = [None, 0.49, 0.495, 0.48, 0.49, 0.49]
        g.trhi = [None, 0.51, 0.51, 0.51, 0.51, 0.51]
        fills = MmFills(g, None, 1, 2880)
        # quotes 0.495/0.505; only bar 3's trade low 0.48 prints through the 0.495 bid (quoted at bar 2)
        buys = [f for f in fills if abs(f[2] - 0.005) < 1e-9 and f[3] == 0.0]
        self.assertTrue(len(fills) >= 1)
        self.assertAlmostEqual(fills[0][1], 0.50 - 0.495 - Fee(0.495, "kalshi_maker"))
        g.wp = [0.5, 0.5, 0.6, 0.6, 0.6, 0.6]
        pulled = MmFills(g, 0.02, 1, 2880)
        self.assertLess(len(pulled), len(fills))

    def test_surface_enters_next_bar_ask_and_settles(self):
        from InPlay.Round2 import Cell, TradeSurface
        g = MakeGame([0.6, 0.6, 0.30, 0.10, 0.10], won=1, pre=0.6, el=[1500, 1560, 1620, 1680, 1740], sd=[-5, -5, -6, -12, -12])
        c = Cell(g, 2, 1)
        tr = TradeSurface([g], {c: {}}, slip=0.0)
        self.assertEqual(len(tr), 1)
        self.assertAlmostEqual(tr[0]["entry"], g.ask[3])
        self.assertEqual(tr[0]["exit"], 1.0)

class TestCiAuditAndTape(unittest.TestCase):
    def test_exact_bound_zero_losses_does_not_collapse(self):
        from InPlay.CiAudit import ClopperPearson
        lo, hi = ClopperPearson(86, 86)
        self.assertAlmostEqual(lo, 0.025 ** (1.0 / 86), places=9)
        self.assertEqual(hi, 1.0)
        self.assertLess(lo - 0.960, 0.0)

    def test_exact_bound_large_n_is_stable(self):
        from InPlay.CiAudit import ClopperPearson
        lo, hi = ClopperPearson(1800, 2000)
        self.assertTrue(0.88 < lo < 0.9 < hi < 0.92)

    def test_day_cluster_bootstrap_not_trade_bootstrap(self):
        from InPlay.CiAudit import Cluster
        # 10 trades share one losing day; one winning trade shares another day.
        tr = [dict(day="a", gid="1", entry=0.5, exit=0, settled=True, maker=False) for _ in range(10)]
        tr += [dict(day="b", gid="2", entry=0.5, exit=1, settled=True, maker=False)]
        (pt, lo, hi), nd = Cluster(tr, "day", "none")
        self.assertEqual(nd, 2)
        self.assertAlmostEqual(pt, -4.5 / 11)
        self.assertAlmostEqual(lo, -0.5)
        self.assertAlmostEqual(hi, 0.5)

    def test_flow_uses_only_past_prints(self):
        import numpy as np
        from InPlay.FlowTape import Flow
        tp = np.array([[90, 0.5, 100, 1], [100, 0.5, 100, -1], [101, 0.5, 10000, 1]], float)
        total, imb = Flow(tp, 100, 30)
        self.assertEqual(total, 200)
        self.assertAlmostEqual(imb, 0)
        tp[2, 3] = -1
        self.assertEqual(Flow(tp, 100, 30), (total, imb))

    def test_real_tape_fill_needs_opposite_taker(self):
        import numpy as np
        from InPlay.TradeTape import TapeFills
        g = MakeGame([0.5] * 5, el=[0, 60, 120, 180, 240])
        g.t = [0, 60, 120, 180, 240]
        g.wp = [0.5] * 5
        # SELL YES (taker no) through our bid at 0.495; then BUY YES at that same low price does not fill a bid.
        tp = np.array([[65, 0.48, 100, -1], [125, 0.48, 100, 1]], float)
        fills = TapeFills(g, tp, None, 1, 2160, "through")
        self.assertEqual(len(fills), 1)
        self.assertAlmostEqual(fills[0][1], 0.5 - 0.495 - Fee(0.495, "kalshi_maker"))

    def test_possession_and_timeout_states_are_asof(self):
        from InPlay.LateCells import States
        game = dict(home_id="h", away_id="a", plays=[
            dict(seq=1, period=4, wall="2026-11-12T02:00:00Z", type="Defensive Rebound", team="h"),
            dict(seq=2, period=4, wall="2026-11-12T02:00:10Z", type="Full Timeout", team="a"),
            dict(seq=3, period=4, wall="2026-11-12T02:00:20Z", type="Jump Shot", team="h", scoring=True, shoot=True),
            dict(seq=4, period=4, wall="2026-11-12T02:00:30Z", type="Official Timeout", team="a")])
        ts, st, h, a = States(game)
        self.assertEqual(st[0], ("h", {"h": 0, "a": 0}))
        self.assertEqual(st[1], ("h", {"h": 0, "a": 1}))
        self.assertEqual(st[2][0], "a")
        self.assertEqual(st[3][1]["a"], 1)

import numpy as np

from InPlay.Round7A import BestLag, ShiftSecondHalf
from InPlay.Round7B import Fills as QueueFills
from InPlay.Round7C import Implied, Trades as SpreadTrades


class TestRoundSeven(unittest.TestCase):
    def test_best_lag_and_second_half_only(self):
        rng = np.random.default_rng(1)
        n = 60
        el = [60.0 * k for k in range(n)]
        wp = list(np.clip(0.5 + np.cumsum(rng.normal(0, 0.03, n)), 0.01, 0.99))
        px = [wp[0]] + wp[:-1]  # PM moves one bar after ESPN
        self.assertEqual(BestLag(wp, px, el), 1)
        sh = ShiftSecondHalf(px, el, 1)
        self.assertEqual(sh[:24], px[:24])  # first half untouched (el < 1440)
        self.assertEqual(sh[30], px[31])

    def test_queue_fill_needs_volume_ahead(self):
        g = MakeGame([0.50] * 10, spread=0.02)
        g.wp = [0.5] * 10
        # one taker sell of 100 at our bid (0.49) in bar 2's window, nothing through
        tape = np.array([[g.t[2] + 10, 0.49, 100.0, -1]])
        self.assertEqual(len(QueueFills(g, tape, "touch", None, 1)), 1)
        self.assertEqual(len(QueueFills(g, tape, "fixed_50", None, 1)), 1)
        self.assertEqual(len(QueueFills(g, tape, "fixed_250", None, 1)), 0)
        self.assertEqual(len(QueueFills(g, tape, "through", None, 1)), 0)
        tape2 = np.array([[g.t[2] + 10, 0.48, 1.0, -1]])  # print through our bid: level exhausted
        self.assertEqual(len(QueueFills(g, tape2, "fixed_1000", None, 1)), 1)

    def test_spread_implied_moneyline(self):
        from statistics import NormalDist
        nd = NormalDist(mu=4.0, sigma=10.0)
        pts = [(k, 1 - nd.cdf(k)) for k in (-5.5, 1.5, 7.5, 12.5)]
        self.assertAlmostEqual(Implied(pts), 1 - nd.cdf(0.5), places=6)
        self.assertIsNone(Implied([(1.5, 0.5)]))

    def test_spread_trade_enters_next_bar(self):
        g = MakeGame([0.50] * 8, spread=0.02)
        imp = [None, 0.60, None, None, None, None, None, None]
        tr = SpreadTrades(g, imp, 0.05, 2)
        self.assertEqual(len(tr), 1)
        self.assertAlmostEqual(tr[0]["entry"], 0.51 + 0.005)  # ask of bar 2 + slip, not bar 1


if __name__ == "__main__":
    unittest.main()
