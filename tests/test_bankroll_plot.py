"""Tests for season-aware bankroll chart framing (no matplotlib required)."""

import datetime as dt
import unittest

from Backtest.BankrollPlot import (
    AsDate,
    DrawdownPct,
    EndOfDayEquity,
    FrameBankroll,
    NbaSeasonLabel,
    SeasonWindows,
    SplitBySeason,
)


class SeasonLabelTests(unittest.TestCase):
    def test_regular_season_spans_calendar_year(self):
        self.assertEqual(NbaSeasonLabel("2024-10-22"), "2024-25")
        self.assertEqual(NbaSeasonLabel("2025-06-22"), "2024-25")
        self.assertEqual(NbaSeasonLabel("2025-10-21"), "2025-26")
        self.assertEqual(NbaSeasonLabel(dt.date(2026, 6, 13)), "2025-26")
        self.assertEqual(NbaSeasonLabel("2026-10-20"), "2026-27")

    def test_bubble_games_stay_in_2019_20(self):
        self.assertEqual(NbaSeasonLabel("2020-03-11"), "2019-20")
        self.assertEqual(NbaSeasonLabel("2020-07-30"), "2019-20")
        self.assertEqual(NbaSeasonLabel("2020-10-11"), "2019-20")
        self.assertEqual(NbaSeasonLabel("2020-12-22"), "2020-21")

    def test_as_date_accepts_iso_and_datetime(self):
        self.assertEqual(AsDate("2025-06-22T19:30:00-04:00"), dt.date(2025, 6, 22))
        self.assertEqual(AsDate(dt.datetime(2025, 6, 22, 23, 0, 0)), dt.date(2025, 6, 22))


class FrameTests(unittest.TestCase):
    def test_single_season_uses_calendar_and_stops_at_last_game(self):
        dates = ["2024-10-22", "2025-01-15", "2025-06-22"]
        equity = [10000.0, 9800.0, 10100.0]
        frame = FrameBankroll(dates, equity, mode="auto")
        self.assertEqual(frame["mode"], "single")
        self.assertEqual(frame["x_kind"], "date")
        self.assertEqual(frame["seasons"], ["2024-25"])
        self.assertEqual(frame["first_date"], "2024-10-22")
        self.assertEqual(frame["last_date"], "2025-06-22")
        self.assertEqual(frame["boundaries"], [])
        xs = [row["x"] for row in frame["rows"]]
        self.assertEqual(xs, ["2024-10-22", "2025-01-15", "2025-06-22"])
        # No invented offseason points between June and October.
        self.assertTrue(all(not row["date"][5:].startswith("07-") and not row["date"][5:].startswith("08-")
                            for row in frame["rows"]))

    def test_single_mode_rejects_two_seasons(self):
        dates = ["2024-10-22", "2025-06-22", "2025-10-21"]
        with self.assertRaises(ValueError):
            FrameBankroll(dates, [1, 2, 3], mode="single")

    def test_multi_season_index_skips_summer_and_marks_boundaries(self):
        dates = [
            "2023-10-24", "2024-01-10", "2024-06-17",
            "2024-10-22", "2025-02-01", "2025-06-22",
            "2025-10-21", "2026-06-13",
        ]
        equity = [100, 110, 105, 105, 90, 80, 80, 70]
        frame = FrameBankroll(dates, equity, series="official", mode="auto")
        self.assertEqual(frame["mode"], "index")
        self.assertEqual(frame["seasons"], ["2023-24", "2024-25", "2025-26"])
        self.assertEqual(frame["last_date"], "2026-06-13")
        xs = [row["x"] for row in frame["rows"]]
        self.assertEqual(xs, list(range(8)))
        # The June-to-October gaps are not represented as x jumps of ~120 days.
        self.assertEqual(xs[-1] - xs[0], 7)
        seasons_at = {row["season"]: row["x"] for row in frame["rows"] if row["x"] == 0 or True}
        bounds = {b["season"]: b["x"] for b in frame["boundaries"] if b["series"] == "official"}
        self.assertEqual(bounds["2024-25"], 3)
        self.assertEqual(bounds["2025-26"], 6)
        self.assertNotIn("2023-24", bounds)

    def test_cut_calendar_removes_offseason_span(self):
        dates = ["2024-06-17", "2024-10-22", "2024-10-24"]
        equity = [100.0, 100.0, 99.0]
        naive_gap = (dt.date(2024, 10, 22) - dt.date(2024, 6, 17)).days
        self.assertGreater(naive_gap, 100)
        frame = FrameBankroll(dates, equity, mode="cut_calendar")
        xs = [row["x"] for row in frame["rows"]]
        self.assertEqual(xs[0], 0)
        # Summer is cut to a one-step boundary, then Oct 22->24 keeps 2 calendar days.
        self.assertEqual(xs[1], 1)
        self.assertEqual(xs[2], 3)
        self.assertLess(xs[2], naive_gap)
        self.assertEqual(len(frame["boundaries"]), 1)
        self.assertEqual(frame["boundaries"][0]["season"], "2024-25")

    def test_split_by_season_ends_at_each_seasons_last_game(self):
        dates = ["2024-10-22", "2025-06-22", "2025-10-21", "2026-06-13"]
        equity = [10, 12, 12, 8]
        parts = SplitBySeason(dates, equity, series="path")
        self.assertEqual(set(parts), {"2024-25", "2025-26"})
        self.assertEqual(parts["2024-25"]["first_date"], "2024-10-22")
        self.assertEqual(parts["2024-25"]["last_date"], "2025-06-22")
        self.assertEqual(parts["2025-26"]["first_date"], "2025-10-21")
        self.assertEqual(parts["2025-26"]["last_date"], "2026-06-13")
        self.assertEqual(parts["2024-25"]["mode"], "single")

    def test_drawdown_and_end_of_day(self):
        self.assertEqual(DrawdownPct([100, 120, 90, 90]), [0.0, 0.0, -0.25, -0.25])
        dates, eq, ser = EndOfDayEquity(
            ["2025-01-01", "2025-01-01", "2025-01-02"],
            [100.0, 95.0, 97.0],
            ["a", "a", "a"],
        )
        self.assertEqual([d.isoformat() for d in dates], ["2025-01-01", "2025-01-02"])
        self.assertEqual(eq, [95.0, 97.0])

    def test_empty_frame(self):
        frame = FrameBankroll([], [])
        self.assertEqual(frame["rows"], [])
        self.assertEqual(frame["seasons"], [])

    def test_season_windows(self):
        windows = SeasonWindows(["2024-10-22", "2025-06-22", "2025-10-21"])
        self.assertEqual(windows[0]["season"], "2024-25")
        self.assertEqual(windows[0]["first"], dt.date(2024, 10, 22))
        self.assertEqual(windows[0]["last"], dt.date(2025, 6, 22))
        self.assertEqual(windows[1]["season"], "2025-26")


if __name__ == "__main__":
    unittest.main()