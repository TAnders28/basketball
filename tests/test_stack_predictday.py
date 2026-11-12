"""Tests for stack_global PredictDay: leak-free weights, schema v2 log, dry-run, default PASS."""

import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from Backtest import Odds, PredictDay, PredictionLog, StackDay
from Backtest.GradePredictions import GradeSeason
from Backtest.Model import EloModel
from Matchups.V3 import Post as PP
from tests.test_v3 import FakeRows, Perturbed


class ConstApproach:
    """Deterministic member that only uses the pregame view (no results)."""

    def __init__(self, name, bias):
        self.name = name
        self.bias = bias

    def BeforeDay(self, day, gap):
        pass

    def Predict(self, view):
        mean = self.bias + (0.5 if view.home > view.away else -0.5)
        return {"margin_mean": mean, "margin_var": 169.0, "p_home": float(0.5 + mean / 80.0)}

    def Update(self, rec):
        pass


def FakeCompleted(n=6, start="2025-11-01"):
    games = []
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    for i in range(n):
        day = (d0 + timedelta(days=i)).isoformat()
        gid = f"0022500{i:03d}"
        home, away = ("AAA", "BBB") if i % 2 == 0 else ("BBB", "AAA")
        games.append({
            "game_id": gid, "game_date": day, "season": 2025,
            "home_team": home, "away_team": away,
            "home_score": 110 + i, "away_score": 100,
        })
    return games


def RecordsFromGames(games):
    from Matchups.V2.Base import GameRecord
    recs = {}
    for g in games:
        recs[g["game_id"]] = GameRecord(g, box=[])
    return recs


class WeightsForDayTests(unittest.TestCase):
    def test_matches_context_stack_on_a_held_day(self):
        good, bad = FakeRows(seed=5), FakeRows(seed=5, noise_model=10.0)
        members = {"good": good, "bad": bad}
        cut = "2025-05-01"
        out, names, _wg, _wc = PP.ContextStack(members, {}, contextual=False, min_rows=200)
        used = next(r for r in out if r["game_date"] == cut)["mix"][0]
        got_names, w, _ctx = PP.WeightsForDay(members, cut, contextual=False, min_rows=200)
        self.assertEqual(list(got_names), list(names))
        np.testing.assert_allclose(w, used, atol=1e-12)

    def test_weights_ignore_outcomes_on_and_after_the_day(self):
        good, bad = FakeRows(seed=5), FakeRows(seed=5, noise_model=10.0)
        cut = "2025-05-01"
        a = PP.WeightsForDay({"good": good, "bad": bad}, cut, min_rows=200)[1]
        b = PP.WeightsForDay({"good": Perturbed(good, cut), "bad": Perturbed(bad, cut)}, cut, min_rows=200)[1]
        np.testing.assert_allclose(a, b, atol=1e-12)


class DecideActionTests(unittest.TestCase):
    def test_pass_without_prices(self):
        d = StackDay.DecideAction(0.80, None)
        self.assertEqual(d["action"], "PASS")
        self.assertEqual(d["reason"], "no prices")
        self.assertIsNone(d["edge"])
        self.assertEqual(d["fee_basis"], Odds.FEE_BASIS)
        self.assertEqual(d["fee"], 0.02)

    def test_pass_when_edge_does_not_clear_fee(self):
        # model 0.56 vs ask 0.55 -> edge = -0.01 after the $0.02 estimate
        d = StackDay.DecideAction(0.56, 0.55)
        self.assertEqual(d["action"], "PASS")
        self.assertEqual(d["reason"], "edge_not_positive")
        self.assertAlmostEqual(d["edge"], -0.01)

    def test_bet_only_when_fee_adjusted_edge_is_positive(self):
        d = StackDay.DecideAction(0.60, 0.55)
        self.assertEqual(d["action"], "BET")
        self.assertAlmostEqual(d["edge"], 0.03)
        self.assertEqual(d["fee_basis"], Odds.FEE_BASIS)


class PredictSlateTests(unittest.TestCase):
    def test_headline_follows_stack_not_elo_and_is_leak_free(self):
        hist = FakeCompleted(8, start="2025-11-01")
        recs = RecordsFromGames(hist)
        slate = [{
            "game_id": "0022500999", "game_date": "2025-11-09", "season": 2025,
            "home_team": "AAA", "away_team": "BBB",
        }]
        # Strong kalman home bias, Elo slightly away; stack should follow kalman once
        # weights have almost no data (equal prior) ... with < min_rows, weights stay 1/K
        # so the mix of a +10 kalman and a weak Elo still leans home.
        approaches = {
            "kalman_team": ConstApproach("kalman_team", 10.0),
            "schedule_fatigue": ConstApproach("schedule_fatigue", 10.0),
        }
        elo_rows = []
        for g in hist:
            elo_rows.append({
                "game_id": g["game_id"], "game_date": g["game_date"], "season": 2025,
                "home": g["home_team"], "away": g["away_team"],
                "margin": g["home_score"] - g["away_score"],
                "home_win": int(g["home_score"] > g["away_score"]),
                "margin_mean": -4.0, "margin_var": 169.0, "p_home": 0.40,
            })
        elo_model = EloModel()
        # An Elo model that would pick the away team on the slate
        elo_model.ratings["AAA"] = 1400
        elo_model.ratings["BBB"] = 1600
        out = StackDay.PredictSlate(
            "2025-11-09", 2025, slate, seasons=[2025], record_from="2025-11-01",
            records=lambda g: recs[g["game_id"]], games=hist, approaches=approaches,
            elo_rows=elo_rows, elo_model=elo_model,
        )
        row = out["rows"][0]
        self.assertEqual(row["recommended"], "AAA")
        self.assertGreater(row["stack_global_home_prob"], 0.5)
        self.assertLess(row["elo_predictday_v1_home_prob"], 0.5)
        self.assertEqual(row["action"], "PASS")
        self.assertEqual(row["reason"], "no prices")
        self.assertEqual(out["games_used"], 8)
        self.assertEqual(out["data_through"], "2025-11-08")

    def test_recommend_uses_stack_when_elo_disagrees(self):
        view = {"game_id": "g", "home_team": "PHI", "away_team": "TOR"}
        picks = {"elo_pick": "away", "better_record": "away", "rest_advantage": None}
        side, prob, tier, agree, against = PredictDay.Recommend(view, picks, 0.64)
        self.assertEqual(side, "home")
        self.assertAlmostEqual(prob, 0.64)
        self.assertEqual(tier, "low")
        self.assertIn("better_record", against)


class PredictionLogSchemaTests(unittest.TestCase):
    def test_old_rows_parse_and_new_rows_carry_shadows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "2026.jsonl")
            v1 = {
                "created_at": "2026-11-12T09:39:07-05:00", "game_id": "0022600227",
                "start_time": "2026-11-12T19:00:00-05:00", "pick": "PHI", "model_prob": 0.6499,
                "market_price": None, "price_time": None, "source": "PredictDay",
                "model_version": "elo_predictday_v1", "game_date": "2026-11-12",
                "home_team": "PHI", "away_team": "TOR",
            }
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(v1) + "\n")
            with mock.patch.object(PredictionLog, "PRED_LOG_DIR", tmp):
                rows = PredictionLog.ReadPredictionLog(2026)
                self.assertEqual(len(rows), 1)
                self.assertEqual(PredictionLog.LogSchema(rows[0]), 1)
                self.assertNotIn("stack_global_home_prob", rows[0])
                PredictionLog.AppendPredictionRows(2026, [{
                    "game_id": "0022600227", "game_date": "2026-11-12", "tip_et": "7:00 pm ET",
                    "home_team": "PHI", "away_team": "TOR",
                    "recommended": "PHI", "recommended_prob": 0.61,
                    "stack_global_home_prob": 0.61, "kalman_team_home_prob": 0.62,
                    "elo_predictday_v1_home_prob": 0.6499,
                    "kalman_team_pick": "PHI", "kalman_team_prob": 0.62,
                    "elo_predictday_v1_pick": "PHI", "elo_predictday_v1_prob": 0.6499,
                    "action": "PASS", "reason": "no prices", "edge": None,
                    "fee": 0.02, "fee_basis": Odds.FEE_BASIS,
                }], created_at="2026-11-12T12:00:00-05:00")
                rows = PredictionLog.ReadPredictionLog(2026)
                self.assertEqual(len(rows), 2)
                with open(path, encoding="utf-8") as fh:
                    self.assertEqual(json.loads(fh.readline())["pick"], "PHI")
                self.assertEqual(rows[0]["model_version"], "elo_predictday_v1")
                self.assertEqual(PredictionLog.LogSchema(rows[1]), 2)
                self.assertEqual(rows[1]["model_version"], PredictionLog.MODEL_VERSION)
                self.assertAlmostEqual(rows[1]["stack_global_home_prob"], 0.61)
                self.assertAlmostEqual(rows[1]["elo_predictday_v1_home_prob"], 0.6499)
                self.assertEqual(rows[1]["action"], "PASS")
                self.assertEqual(rows[1]["fee_basis"], Odds.FEE_BASIS)

    def test_grader_still_reads_mixed_schema_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            season_dir = os.path.join(tmp, "season")
            os.makedirs(season_dir)
            with open(os.path.join(season_dir, "games_2026.json"), "w", encoding="utf-8") as fh:
                json.dump([{
                    "game_id": "g1", "game_date": "2026-11-08",
                    "home_team": "MIN", "away_team": "UTA",
                    "home_score": 115, "away_score": 100,
                }], fh)
            with mock.patch.object(PredictionLog, "PRED_LOG_DIR", tmp), \
                 mock.patch("Backtest.GradePredictions.SEASON_DIR", season_dir):
                # v1 row written by hand
                with open(PredictionLog.LogPath(2026), "w", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "created_at": "2026-11-08T12:00:00-05:00", "game_id": "g1",
                        "start_time": "2026-11-08T19:00:00-05:00", "pick": "MIN",
                        "model_prob": 0.9, "market_price": None, "price_time": None,
                        "source": "PredictDay", "model_version": "elo_predictday_v1",
                        "game_date": "2026-11-08", "home_team": "MIN", "away_team": "UTA",
                    }) + "\n")
                summary = GradeSeason(2026)
                self.assertEqual(summary["graded"], 1)
                self.assertTrue(summary["grades"]["g1"]["correct"])


class DryRunTests(unittest.TestCase):
    def test_dry_run_does_not_append_or_write_default_picks(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "predictions")
            pred_dir = os.path.join(tmp, "picks")
            os.makedirs(log_dir)
            os.makedirs(pred_dir)
            existing = os.path.join(log_dir, "2026.jsonl")
            with open(existing, "w", encoding="utf-8") as fh:
                fh.write("{}\n")
            with open(existing, encoding="utf-8") as fh:
                before = fh.read()
            slate = [{
                "game_id": "0022600227", "game_date": "2026-11-12",
                "home_team": "PHI", "away_team": "TOR",
                "tip_et": "7:00 pm ET", "season_type": "Regular Season",
                "home_moneyline": None, "away_moneyline": None, "line": "-",
            }]
            stack_row = {
                "game_id": "0022600227", "game_date": "2026-11-12",
                "home_team": "PHI", "away_team": "TOR",
                "stack_global_home_prob": 0.61, "kalman_team_home_prob": 0.62,
                "elo_predictday_v1_home_prob": 0.6499,
                "recommended_side": "home", "recommended": "PHI", "recommended_prob": 0.61,
                "kalman_team_pick": "PHI", "kalman_team_prob": 0.62,
                "elo_predictday_v1_pick": "PHI", "elo_predictday_v1_prob": 0.6499,
                "margin_mean": 3.0, "margin_var": 169.0,
                "action": "PASS", "reason": "no prices", "edge": None,
                "fee": 0.02, "fee_basis": Odds.FEE_BASIS,
            }
            features = mock.Mock()
            features.Team.return_value = mock.Mock(wins=5, losses=5)
            elo = EloModel()
            with mock.patch.object(PredictionLog, "PRED_LOG_DIR", log_dir), \
                 mock.patch.object(PredictDay, "PRED_DIR", pred_dir), \
                 mock.patch.object(PredictDay, "FetchSlate", return_value=slate), \
                 mock.patch.object(PredictDay, "BuildState", return_value=(features, elo, [])), \
                 mock.patch.object(PredictDay, "PreGameView", return_value={
                     "home_rest_days": 1, "away_rest_days": 1,
                     "home_avg_margin": 1.0, "away_avg_margin": -1.0,
                     "home_games": 10, "away_games": 10,
                     "home_win_pct": 0.5, "away_win_pct": 0.5,
                     "model_home_prob": 0.65, "model_away_prob": 0.35,
                 }), \
                 mock.patch.object(StackDay, "PredictSlate", return_value={
                     "rows": [stack_row], "names": ["kalman_team", "elo_phase1"],
                     "weights": [0.7, 0.3], "data_through": "2026-11-11", "games_used": 12,
                     "seasons": [2024, 2025, 2026],
                 }):
                payload = PredictDay.Run(PredictDay.ParseArgs([
                    "--date", "2026-11-12", "--season", "2026", "--dry-run",
                ]))
            self.assertTrue(payload["dry_run"])
            self.assertIsNone(payload["log_path"])
            with open(existing, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), before)
            self.assertFalse(os.path.exists(os.path.join(pred_dir, "picks_2026-11-12.json")))
            self.assertEqual(payload["games"][0]["recommended"], "PHI")
            self.assertEqual(payload["games"][0]["action"], "PASS")
            self.assertEqual(payload["games"][0]["model_version"], PredictionLog.MODEL_VERSION)

    def test_wet_run_appends_schema_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "predictions")
            pred_dir = os.path.join(tmp, "picks")
            os.makedirs(log_dir)
            os.makedirs(pred_dir)
            slate = [{
                "game_id": "0022600227", "game_date": "2026-11-12",
                "home_team": "PHI", "away_team": "TOR",
                "tip_et": "7:00 pm ET", "season_type": "Regular Season",
                "home_moneyline": None, "away_moneyline": None, "line": "-",
            }]
            stack_row = {
                "game_id": "0022600227", "game_date": "2026-11-12",
                "home_team": "PHI", "away_team": "TOR",
                "stack_global_home_prob": 0.61, "kalman_team_home_prob": 0.62,
                "elo_predictday_v1_home_prob": 0.6499,
                "recommended_side": "home", "recommended": "PHI", "recommended_prob": 0.61,
                "kalman_team_pick": "PHI", "kalman_team_prob": 0.62,
                "elo_predictday_v1_pick": "PHI", "elo_predictday_v1_prob": 0.6499,
                "margin_mean": 3.0, "margin_var": 169.0,
                "action": "PASS", "reason": "no prices", "edge": None,
                "fee": 0.02, "fee_basis": Odds.FEE_BASIS,
            }
            features = mock.Mock()
            features.Team.return_value = mock.Mock(wins=5, losses=5)
            elo = EloModel()
            with mock.patch.object(PredictionLog, "PRED_LOG_DIR", log_dir), \
                 mock.patch.object(PredictDay, "PRED_DIR", pred_dir), \
                 mock.patch.object(PredictDay, "FetchSlate", return_value=slate), \
                 mock.patch.object(PredictDay, "BuildState", return_value=(features, elo, [])), \
                 mock.patch.object(PredictDay, "PreGameView", return_value={
                     "home_rest_days": 1, "away_rest_days": 1,
                     "home_avg_margin": 1.0, "away_avg_margin": -1.0,
                     "home_games": 10, "away_games": 10,
                     "home_win_pct": 0.5, "away_win_pct": 0.5,
                     "model_home_prob": 0.65, "model_away_prob": 0.35,
                 }), \
                 mock.patch.object(StackDay, "PredictSlate", return_value={
                     "rows": [stack_row], "names": ["kalman_team", "elo_phase1"],
                     "weights": [0.7, 0.3], "data_through": "2026-11-11", "games_used": 12,
                     "seasons": [2024, 2025, 2026],
                 }):
                payload = PredictDay.Run(PredictDay.ParseArgs([
                    "--date", "2026-11-12", "--season", "2026",
                ]))
            self.assertFalse(payload["dry_run"])
            with open(os.path.join(log_dir, "2026.jsonl"), encoding="utf-8") as fh:
                logged = [json.loads(l) for l in fh]
            self.assertEqual(len(logged), 1)
            self.assertEqual(logged[0]["log_schema"], 2)
            self.assertEqual(logged[0]["model_version"], PredictionLog.MODEL_VERSION)
            self.assertAlmostEqual(logged[0]["elo_predictday_v1_home_prob"], 0.6499)
            self.assertEqual(logged[0]["action"], "PASS")
            self.assertTrue(os.path.exists(os.path.join(pred_dir, "picks_2026-11-12.json")))


if __name__ == "__main__":
    unittest.main()