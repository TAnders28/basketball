import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Tools import Data
from Tools.SlipCheck import LegFromMarket, Ticket, GameKey
from Tools.LiveView import Swing
from Tools.Board import BuildBoard


class OddsMath(unittest.TestCase):
    def test_american_and_novig(self):
        self.assertAlmostEqual(Data.AmericanToProb(-150), 0.6)
        self.assertAlmostEqual(Data.AmericanToProb(150), 0.4)
        self.assertIsNone(Data.AmericanToProb(50))
        fair = Data.NoVig([0.84, 0.17])
        self.assertAlmostEqual(sum(fair), 1.0)
        self.assertAlmostEqual(fair[0], 0.84 / 1.01)
        self.assertIsNone(Data.NoVig([0.5, None]))

    def test_edge_cents_and_money(self):
        self.assertEqual(Data.EdgeCents(0.60, 0.50), 8.0)
        self.assertIsNone(Data.EdgeCents(None, 0.5))
        self.assertEqual(Data.Money(-0.16), "-$0.16")
        self.assertEqual(Data.FmtAmerican(0.5), "-100")

    def test_alias_and_datecode_and_et(self):
        self.assertEqual(Data.Std("NO"), "NOP")
        self.assertEqual(Data.Std("gs"), "GSW")
        self.assertEqual(Data.DateCode("2026-11-14"), "26NOV14")
        self.assertEqual(Data.IsoEt("2026-11-15T00:00Z"), "2026-11-14T19:00:00-05:00")
        self.assertIsNone(Data.ToEt("garbage"))


class Slip(unittest.TestCase):
    def test_moneyline_leg_uses_novig_and_no_side(self):
        a = {"ticker": "KXNBAGAME-26NOV14DENNOP-DEN", "yes_ask": 0.84, "yes_bid": 0.83}
        b = {"ticker": "KXNBAGAME-26NOV14DENNOP-NOP", "yes_ask": 0.17, "yes_bid": 0.16}
        leg = LegFromMarket(a, "yes", b)
        self.assertAlmostEqual(leg["fair"], 0.84 / 1.01)
        leg_no = LegFromMarket(a, "no", b)
        self.assertAlmostEqual(leg_no["ask"], 0.17)
        self.assertAlmostEqual(leg_no["fair"], 1 - 0.84 / 1.01)
        self.assertEqual(LegFromMarket(None)["error"], "market not found")

    def test_ticket_math_and_same_game_flag(self):
        legs = [{"fair": 0.5, "ask": 0.52, "game": "26NOV14DENNOP"}, {"fair": 0.4, "ask": 0.42, "game": "26NOV14DENNOP"}]
        tk = Ticket(legs, 0.25, 10)
        self.assertAlmostEqual(tk["fair_prob"], 0.2)
        self.assertEqual(tk["cost"], 2.5)
        self.assertEqual(tk["fee_total"], 0.2)
        self.assertEqual(tk["price_minus_fair_c"], 5.0)
        self.assertEqual(tk["same_game_keys"], ["26NOV14DENNOP"])
        self.assertAlmostEqual(tk["value_vs_fair_total"], 10 * (0.2 - 0.25 - 0.02))
        self.assertTrue(tk["independence_assumed"])

    def test_ticket_missing_leg_is_graceful(self):
        tk = Ticket([{"fair": None, "ask": None}], None, 3)
        self.assertIsNone(tk["fair_prob"])
        self.assertIsNone(tk["cost"])
        self.assertEqual(GameKey("KXNBAPTS-26NOV14DENNOP-DENJMURRAY27-25"), "26NOV14DENNOP")


class Live(unittest.TestCase):
    def test_swing(self):
        self.assertEqual(Swing(0.62, 0.55), 7.0)
        self.assertIsNone(Swing(None, 0.5))


class BoardSchema(unittest.TestCase):
    def test_board_nulls_and_tested_false(self):
        espn = {"espn_id": "1", "start": "2026-11-14T19:00:00-05:00", "state": "pre", "detail": "7:00 PM",
                "home_info": {"score": 0}, "away_info": {"score": 0}}
        g = {"espn": espn, "home": "NOP", "away": "DEN", "kalshi": None, "move_home": None, "book": None, "predictor": None,
             "wp": None, "injuries_official": [], "injuries_espn": [], "season_series": [], "form": {"NOP": None, "DEN": None},
             "stats": {}, "h2h": [], "trends": [],
             "props": [{"player": "X", "stat": "points", "k": 20, "yes_ask": 0.5, "yes_bid": 0.48, "ticker": "T"}]}
        b = BuildBoard("2026-11-14", {"games": [g], "errors": []}, {}, None)
        gg = b["games"][0]
        self.assertEqual(b["sport"], "NBA")
        self.assertIsNone(gg["market"]["home_yes_ask"])
        self.assertIsNone(gg["model"]["home_win_prob"])
        self.assertFalse(gg["model"]["tested"])
        self.assertIsNone(gg["edge_home_c"])
        self.assertIsNone(gg["props"][0]["edge_c"])
        self.assertFalse(gg["props"][0]["tested"])
        self.assertIsNone(gg["context"]["rest_days"]["NOP"])

    def test_board_edge_formula(self):
        espn = {"espn_id": "1", "start": None, "state": "pre", "detail": None, "home_info": {}, "away_info": {}}
        k = {"home": {"yes_ask": 0.40}, "away": {"yes_ask": 0.62}, "home_fair_ask": 0.39, "away_fair_ask": 0.61}
        g = {"espn": espn, "home": "MIA", "away": "DET", "kalshi": k, "move_home": None, "book": None, "predictor": None,
             "wp": None, "injuries_official": [], "injuries_espn": [], "season_series": [], "form": {}, "stats": {}, "h2h": [],
             "trends": [], "props": []}
        b = BuildBoard("2026-11-14", {"games": [g], "errors": []}, {("MIA", "DET"): 0.45}, {"data_through": "2026-11-12"})
        self.assertEqual(b["games"][0]["edge_home_c"], 3.0)
        self.assertEqual(b["games"][0]["edge_away_c"], -9.0)


class Atomic(unittest.TestCase):
    def test_write_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            Data.WriteAtomic(p, "{}")
            self.assertEqual(p.read_text(), "{}")
            self.assertFalse((Path(d) / "x.json.tmp").exists())


class Injury(unittest.TestCase):
    def test_injuries_for_game_filters_matchup(self):
        rep = {"source": "NBA official injury report PDF", "report_slot_et": "2026-11-14T18:00-05:00",
               "rows": [{"matchup": "DEN@NOP", "team": "DEN", "player": "A", "status": "Out", "reason": "x"},
                        {"matchup": "DAL@CLE", "team": "DAL", "player": "B", "status": "Out", "reason": "y"}]}
        out = Data.InjuriesForGame(rep, "DEN", "NOP")
        self.assertEqual([r["player"] for r in out], ["A"])
        self.assertEqual(Data.InjuriesForGame(None, "DEN", "NOP"), [])


if __name__ == "__main__":
    unittest.main()