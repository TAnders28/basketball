"""Shared framing for bankroll and equity-curve charts.

Tim's rule (2026-11-13): a bankroll chart must cover exactly one season, from
its first game to its last, with no flat offseason stretches. A multi-season
chart must skip the summer gaps (put the x-axis on a game or trading-day index
with a marker at each season boundary, or cut out the offseason dates) and end
at the last game of the final season.

This module is matplotlib-free. Callers can plot from FrameBankroll output, or
use PlotBankroll if matplotlib is installed.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

# 2019-20 bubble restart through the Finals (Orlando). Same NBA season, not a
# new one, even though games were played in July-October 2020.
BUBBLE_START = dt.date(2020, 7, 30)
BUBBLE_END = dt.date(2020, 10, 11)

# Longer than All-Star / Christmas breaks; shorter than a normal summer and
# the short 2020 post-bubble offseason (~72 days). Used only to describe gaps,
# not to assign season labels.
OFFSEASON_GAP_DAYS = 56


def AsDate(value) -> dt.date:
    """Coerce a date, datetime, pandas Timestamp, numpy datetime64, or ISO string."""
    if value is None:
        raise TypeError("date value is None")
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    if hasattr(value, "date") and not isinstance(value, (str, bytes)):
        try:
            out = value.date()
            if isinstance(out, dt.date):
                return out
        except Exception:
            pass
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" in text:
        return dt.datetime.fromisoformat(text).date()
    return dt.date.fromisoformat(text[:10])


def NbaSeasonLabel(value) -> str:
    """Return 'YYYY-YY' for an NBA game date (2019-20 bubble kept in 2019-20)."""
    day = AsDate(value)
    if BUBBLE_START <= day <= BUBBLE_END:
        return "2019-20"
    if day.month >= 9:
        return f"{day.year}-{str(day.year + 1)[2:]}"
    return f"{day.year - 1}-{str(day.year)[2:]}"


def DrawdownPct(equity) -> list[float]:
    """Running drawdown as a negative fraction of the running peak (0 if flat/up)."""
    out = []
    peak = None
    for raw in equity:
        val = float(raw)
        peak = val if peak is None or val > peak else peak
        if peak == 0:
            out.append(0.0)
        else:
            dd = val / peak - 1.0
            out.append(dd if dd < 0 else 0.0)
    return out


def EndOfDayEquity(dates, equity, series=None):
    """Keep the last equity value on each (series, date), preserving series order."""
    last = {}
    order = []
    n = len(dates)
    if len(equity) != n:
        raise ValueError("dates and equity must be the same length")
    if series is None:
        series = [None] * n
    elif len(series) != n:
        raise ValueError("series must be the same length as dates")
    for i in range(n):
        key = (series[i], AsDate(dates[i]))
        if key not in last:
            order.append(key)
        last[key] = float(equity[i])
    out_dates = [k[1] for k in order]
    out_eq = [last[k] for k in order]
    out_series = [k[0] for k in order]
    return out_dates, out_eq, out_series


def SeasonWindows(dates) -> list[dict]:
    """Unique seasons in order, each with first and last game date."""
    windows = []
    index = {}
    for value in dates:
        day = AsDate(value)
        label = NbaSeasonLabel(day)
        if label not in index:
            index[label] = len(windows)
            windows.append({"season": label, "first": day, "last": day})
        else:
            slot = windows[index[label]]
            if day < slot["first"]:
                slot["first"] = day
            if day > slot["last"]:
                slot["last"] = day
    return windows


def FrameBankroll(dates, equity, series=None, mode="auto"):
    """Build a chart frame that does not leave flat offseason stretches.

    mode:
      - 'single': calendar dates, first game to last game of the one season.
      - 'index': trading-day (point) index with a marker at each new season.
      - 'cut_calendar': intra-season calendar spacing, offseason days removed.
      - 'auto': 'single' if every point is in one season, else 'index'.

    Returns a dict with rows (date, series, equity, drawdown_pct, season, x,
    x_kind), season windows, boundary markers, and the mode actually used.
    """
    dates = list(dates)
    equity = [float(v) for v in equity]
    if len(dates) != len(equity):
        raise ValueError("dates and equity must be the same length")
    if series is None:
        series_vals = ["bankroll"] * len(dates)
    elif isinstance(series, (str, bytes)):
        series_vals = [str(series)] * len(dates)
    else:
        series_vals = [str(v) for v in series]
        if len(series_vals) != len(dates):
            raise ValueError("series must be the same length as dates")

    points = sorted(
        (
            (AsDate(d), i, series_vals[i], equity[i])
            for i, d in enumerate(dates)
        ),
        key=lambda row: (row[2], row[0], row[1]),
    )
    if not points:
        return {
            "rows": [],
            "seasons": [],
            "boundaries": [],
            "mode": "single" if mode in ("auto", "single") else mode,
            "x_kind": "date" if mode != "index" else "index",
        }

    seasons = []
    seen = set()
    for day, _i, _s, _e in points:
        label = NbaSeasonLabel(day)
        if label not in seen:
            seen.add(label)
            seasons.append(label)

    if mode == "auto":
        mode_used = "single" if len(seasons) == 1 else "index"
    else:
        if mode not in ("single", "index", "cut_calendar"):
            raise ValueError(f"unknown mode: {mode}")
        if mode == "single" and len(seasons) > 1:
            raise ValueError(
                "mode='single' needs one season; got "
                + ", ".join(seasons)
                + ". Use mode='index' or 'cut_calendar', or split by season."
            )
        mode_used = mode

    # Per-series running drawdown on the sorted path (no offseason fill).
    by_series_eq = defaultdict(list)
    draft = []
    for day, _i, ser, eq in points:
        by_series_eq[ser].append(eq)
        draft.append((day, ser, eq, NbaSeasonLabel(day)))
    dd_iters = {ser: iter(DrawdownPct(vals)) for ser, vals in by_series_eq.items()}

    rows = []
    if mode_used == "index":
        # Independent index per series so several series can share a chart.
        counters = defaultdict(int)
        first_x = {}
        for day, ser, eq, label in draft:
            x = counters[ser]
            counters[ser] += 1
            if (ser, label) not in first_x:
                first_x[(ser, label)] = x
            rows.append({
                "date": day.isoformat(),
                "series": ser,
                "equity": eq,
                "drawdown_pct": next(dd_iters[ser]),
                "season": label,
                "x": x,
                "x_kind": "index",
            })
        boundaries = []
        for ser, label in first_x:
            if label == seasons[0]:
                continue
            boundaries.append({
                "series": ser,
                "season": label,
                "x": first_x[(ser, label)],
                "date": next(
                    r["date"] for r in rows
                    if r["series"] == ser and r["season"] == label
                ),
            })
        x_kind = "index"
    elif mode_used == "cut_calendar":
        # Map each in-season date to a compacted day count that skips summers.
        unique_days = sorted({day for day, _ser, _eq, _label in draft})
        compact = {}
        cursor = 0
        compact[unique_days[0]] = 0
        for prev, cur in zip(unique_days, unique_days[1:]):
            gap = (cur - prev).days
            if gap >= OFFSEASON_GAP_DAYS:
                cursor += 1
            else:
                cursor += gap
            compact[cur] = cursor
        first_x = {}
        for day, ser, eq, label in draft:
            x = compact[day]
            if (ser, label) not in first_x:
                first_x[(ser, label)] = x
            rows.append({
                "date": day.isoformat(),
                "series": ser,
                "equity": eq,
                "drawdown_pct": next(dd_iters[ser]),
                "season": label,
                "x": x,
                "x_kind": "cut_calendar",
            })
        boundaries = []
        for ser, label in first_x:
            if label == seasons[0]:
                continue
            boundaries.append({
                "series": ser,
                "season": label,
                "x": first_x[(ser, label)],
                "date": next(
                    r["date"] for r in rows
                    if r["series"] == ser and r["season"] == label
                ),
            })
        x_kind = "cut_calendar"
    else:
        for day, ser, eq, label in draft:
            rows.append({
                "date": day.isoformat(),
                "series": ser,
                "equity": eq,
                "drawdown_pct": next(dd_iters[ser]),
                "season": label,
                "x": day.isoformat(),
                "x_kind": "date",
            })
        boundaries = []
        x_kind = "date"

    return {
        "rows": rows,
        "seasons": seasons,
        "windows": SeasonWindows([p[0] for p in points]),
        "boundaries": boundaries,
        "mode": mode_used,
        "x_kind": x_kind,
        "first_date": min(p[0] for p in points).isoformat() if points else None,
        "last_date": max(p[0] for p in points).isoformat() if points else None,
    }


def SplitBySeason(dates, equity, series=None) -> dict:
    """Return FrameBankroll(..., mode='single') for each season, keyed by label."""
    dates = list(dates)
    equity = list(equity)
    if series is None:
        series_vals = ["bankroll"] * len(dates)
    elif isinstance(series, (str, bytes)):
        series_vals = [str(series)] * len(dates)
    else:
        series_vals = [str(v) for v in series]
    buckets = defaultdict(lambda: {"dates": [], "equity": [], "series": []})
    for i, value in enumerate(dates):
        label = NbaSeasonLabel(value)
        buckets[label]["dates"].append(value)
        buckets[label]["equity"].append(equity[i])
        buckets[label]["series"].append(series_vals[i])
    return {
        label: FrameBankroll(b["dates"], b["equity"], b["series"], mode="single")
        for label, b in buckets.items()
    }


def TidyRows(frame) -> list[dict]:
    """Rows ready for a Graphtard-style tidy CSV (date, series, equity, drawdown_pct, ...)."""
    return [dict(row) for row in frame["rows"]]


def PlotBankroll(frame, path=None, title=None, ylabel="Bankroll", ax=None,
                 figsize=(11, 6.5)):
    """Optional matplotlib renderer. Returns the Axes. Saves PNG if path is set."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter, AutoDateLocator
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed in this interpreter; "
            "write the tidy CSV and render with a venv that has it"
        ) from exc

    created = ax is None
    fig = None
    if created:
        fig, ax = plt.subplots(figsize=figsize)

    groups = defaultdict(list)
    for row in frame["rows"]:
        groups[row["series"]].append(row)

    kind = frame["x_kind"]
    for ser, rows in groups.items():
        xs = []
        ys = []
        for row in rows:
            if kind == "date":
                xs.append(AsDate(row["x"]))
            else:
                xs.append(row["x"])
            ys.append(row["equity"])
        ax.plot(xs, ys, lw=1.8, label=ser)

    for bound in frame.get("boundaries") or []:
        ax.axvline(bound["x"], color="#555555", lw=1, ls="--", alpha=0.7)
        ymax = ax.get_ylim()[1]
        ax.text(
            bound["x"], ymax, f"  {bound['season']}",
            rotation=90, va="top", ha="left", fontsize=9, color="#333333",
        )

    if kind == "date":
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
        ax.set_xlabel("Game date")
        if frame.get("first_date") and frame.get("last_date"):
            ax.set_xlim(AsDate(frame["first_date"]), AsDate(frame["last_date"]))
    elif kind == "index":
        ax.set_xlabel("Trading-day index (offseason days omitted; dashed line = new season)")
    else:
        ax.set_xlabel("Compacted calendar days (summer gaps cut; dashed line = new season)")

    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if len(groups) > 1:
        ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    if fig is not None:
        fig.tight_layout()
        if path:
            fig.savefig(path, dpi=150)
    return ax