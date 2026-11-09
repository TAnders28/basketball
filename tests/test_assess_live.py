"""AssessLive writer tests using LiveOdds fixtures. No network / no polling."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from Backtest import AssessLive
from Backtest.Odds import FEE_BASIS
from LiveOdds.Report import BuildReport
from LiveOdds.Time import MAX_GAME_STATE_AGE_SECONDS, MAX_LIVE_QUOTE_AGE_SECONDS

OBSERVED = datetime(2026, 11, 9, 1, 42, 35, tzinfo=timezone.utc)
FRESH = OBSERVED - timedelta(seconds=1)


def _ms(moment):
    return str(int(moment.timestamp() * 1000))


def _raw(**overrides):
    market = {
        "sportsMarketType": "moneyline",
        "slug": "nba-uta-min-2026-11-08",
        "question": "Jazz vs. Timberwolves",
        "conditionId": "0xabc",
        "outcomes": json.dumps(["Jazz", "Timberwolves"]),
        "clobTokenIds": json.dumps(["111", "222"]),
        "gameStartTime": "2026-11-08 23:00:00+00",
        "closed": False,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "archived": False,
        "umaResolutionStatus": None,
        "description": "final score including any overtime periods.",
        "resolutionSource": "https://www.nba.com/",
        "feesEnabled": True,
        "feeType": "sports_fees_v3",
        "feeSchedule": {"exponent": 1, "rate": 0.05, "takerOnly": True, "rebateRate": 0.15},
        "makerBaseFee": 1000,
        "takerBaseFee": 1000,
        "bestBid": 0.40,
        "bestAsk": 0.42,
        "outcomePrices": json.dumps(["0.41", "0.59"]),
        "events": [{
            "eventDate": "2026-11-08", "score": "63-68", "period": "Q3",
            "elapsed": "06:06", "live": True, "ended": False,
            "updatedAt": "2026-11-09T01:37:50Z", "gameId": "example-game",
        }],
    }
    market.update(overrides)
    return market


def _book(token, bids, asks, moment=FRESH):
    return {
        "body": {
            "market": "0xabc", "asset_id": token, "timestamp": _ms(moment),
            "bids": [{"price": p, "size": s} for p, s in bids],
            "asks": [{"price": p, "size": s} for p, s in asks],
            "min_order_size": "5", "tick_size": "0.01",
        },
        "observed_at": OBSERVED,
        "url": f"https://clob.polymarket.com/book?token_id={token}",
    }


def _bundle(meta=None, books=None):
    scoreboard = {
        "body": {
            "meta": {"time": meta or "2026-11-08 20:42:32.4232"},
            "scoreboard": {
                "gameDate": "2026-11-08",
                "games": [{
                    "gameId": "0022600123",
                    "gameStatus": 2,
                    "gameStatusText": "Q3 6:06",
                    "period": 3,
                    "gameClock": "PT06M06.00S",
                    "regulationPeriods": 4,
                    "homeTeam": {"teamTricode": "MIN", "teamName": "Timberwolves", "score": 68},
                    "awayTeam": {"teamTricode": "UTA", "teamName": "Jazz", "score": 63},
                }],
            },
        },
        "observed_at": OBSERVED,
        "url": "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
    }
    return {
        "date": "2026-11-08",
        "live_sample": False,
        "polymarket_markets": [_raw()],
        "books": books or {
            "111": _book("111", [("0.40", "100")], [("0.42", "80")]),
            "222": _book("222", [("0.58", "80")], [("0.60", "100")]),
        },
        "fees": {
            "111": {"body": {"base_fee": 1000}, "observed_at": OBSERVED},
            "222": {"body": {"base_fee": 1000}, "observed_at": OBSERVED},
        },
        "scoreboard": scoreboard,
        "play_by_play": {},
        "kalshi_markets": [],
        "kalshi_books": {},
        "kalshi_series": {},
        "pregame_model": None,
    }


class AssessLiveTests(unittest.TestCase):
    def test_defaults_match_liveodds_thresholds(self):
        self.assertEqual(MAX_LIVE_QUOTE_AGE_SECONDS, 30)
        self.assertEqual(MAX_GAME_STATE_AGE_SECONDS, 45)

    def test_always_pass_no_live_model_and_appends_jsonl(self):
        report = BuildReport(_bundle(meta="2026-11-09T01:42:30+00:00"))
        with tempfile.TemporaryDirectory() as tmp:
            result = AssessLive.Assess(
                "2026-11-08", out_root=tmp, report=report,
                max_quote_age=30, max_state_age=45,
            )
            path = result["jsonl"]
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(result["latest_md"]))
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().strip().splitlines()
            self.assertGreaterEqual(len(lines), 1)
            for line in lines:
                row = json.loads(line)
                self.assertEqual(row["action"], "PASS")
                self.assertEqual(row["reason"], "no live model")
                self.assertEqual(row["fee_basis"], FEE_BASIS)
                self.assertIsNone(row["model_prob"])
                self.assertIsNone(row["edge"])
                self.assertIn("quote_qualified", row)
            AssessLive.Assess("2026-11-08", out_root=tmp, report=report)
            with open(path, encoding="utf-8") as fh:
                lines2 = fh.read().strip().splitlines()
            self.assertEqual(len(lines2), 2 * len(lines))

    def test_quote_qualified_false_when_unaligned(self):
        report = BuildReport(_bundle())
        rows = AssessLive.RowsFromReport(report, 30, 45)
        self.assertTrue(rows)
        self.assertFalse(any(r["quote_qualified"] for r in rows))
        self.assertTrue(all(r["action"] == "PASS" for r in rows))

    def test_cli_overrides_do_not_change_module_defaults(self):
        self.assertEqual(MAX_LIVE_QUOTE_AGE_SECONDS, 30)
        with tempfile.TemporaryDirectory() as tmp:
            report = BuildReport(_bundle(meta="2026-11-09T01:42:30+00:00"))
            result = AssessLive.Assess(
                "2026-11-08", out_root=tmp, report=report,
                max_quote_age=5, max_state_age=5,
            )
            self.assertEqual(result["max_quote_age"], 5)
            self.assertEqual(result["max_state_age"], 5)
            self.assertEqual(MAX_LIVE_QUOTE_AGE_SECONDS, 30)
            self.assertEqual(MAX_GAME_STATE_AGE_SECONDS, 45)


if __name__ == "__main__":
    unittest.main()
