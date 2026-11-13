"""Replay tune-selected in-play rules to season-framed bankroll (equity) curves.

Uses the honest no-shift panel (games_obj_s0.pkl). Does not touch the frozen
paper spec or paper_registry.json. Writes tidy CSV + a numbers check; draft
PNGs are rendered separately if matplotlib is available.

.venv\\Scripts\\python.exe -m InPlay.RenderBankroll
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from Matchups.Cpu import LimitThreads, LowerPriority, MaxWorkers, WorkerInit

LimitThreads()
LowerPriority()

from Backtest.BankrollPlot import EndOfDayEquity, FrameBankroll, SplitBySeason, TidyRows
from InPlay.Engine import RunGame, TradePnl
from InPlay.Grid import EngineFamily
from InPlay.Load import LoadGames, SplitOf

REPO = Path(__file__).resolve().parents[1]
GAMES_PATH = REPO / "data" / "inplay" / "games_obj_s0.pkl"
SCORE_PATH = REPO / "data" / "inplay" / "results" / "score.json"
OUT_DIR = Path(r"E:\AgentDocs\Basketbot\2026-11-12_inplay-swings")
BASE = "2026-11-12_inplay-swings_v2"
FEE_BASES = ("tim_est_0.02", "kalshi_taker")
START_EQUITY = 0.0  # 1-contract book; equity is cumulative P&L in dollars
G = None


def InitWorker():
    global G
    WorkerInit()
    G = LoadGames(str(GAMES_PATH))


def RunOne(job):
    fam, params, venue, exec_mode, trial_id = job
    trades = []
    n_games = defaultdict(int)
    for g in G[venue]:
        sp = SplitOf(g.season)
        if sp is None:
            continue
        n_games[sp] += 1
        for tr in RunGame(EngineFamily(fam), params, g, exec_mode=exec_mode):
            tr["split"] = sp
            tr["venue"] = venue
            tr["trial_id"] = trial_id
            tr["family"] = fam
            tr["exec"] = exec_mode
            trades.append(tr)
    return trial_id, trades, dict(n_games)


def LoadJobs():
    score = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
    jobs = []
    meta = {}
    for tid, row in score.items():
        if row.get("tag") != "selected_on_tune":
            continue
        if row.get("exec") != "taker":
            continue
        jobs.append((row["family"], row["params"], row["venue"], row["exec"], tid))
        meta[tid] = row
    return jobs, meta, score


def DailyEquity(trades, basis):
    """End-of-day cumulative P&L, chronological, starting at START_EQUITY."""
    ordered = sorted(trades, key=lambda tr: (str(tr["day"])[:10], tr["gid"], tr.get("el") or 0))
    running = START_EQUITY
    dates, equity, splits, seasons = [], [], [], []
    pnl_by_split = defaultdict(float)
    n_by_split = defaultdict(int)
    for tr in ordered:
        running += TradePnl(tr, basis)
        day = str(tr["day"])[:10]
        dates.append(day)
        equity.append(running)
        splits.append(tr["split"])
        seasons.append(tr.get("season"))
        pnl_by_split[tr["split"]] += TradePnl(tr, basis)
        n_by_split[tr["split"]] += 1
    d, e, _ = EndOfDayEquity(dates, equity, ["x"] * len(dates)) if dates else ([], [], [])
    # map end-of-day back to the last split that day
    last_split = {}
    for day, sp in zip(dates, splits):
        last_split[day] = sp
    eod_splits = [last_split[day.isoformat()] for day in d]
    return {
        "dates": d,
        "equity": e,
        "splits": eod_splits,
        "total": running if ordered else 0.0,
        "n_trades": len(ordered),
        "pnl_by_split": dict(pnl_by_split),
        "n_by_split": dict(n_by_split),
    }


def Compare(tid, basis, split, replay, published):
    pub = ((published.get("res") or {}).get(split) or {}).get(basis) or {}
    if not pub:
        return {"trial_id": tid, "fee_basis": basis, "split": split, "status": "no_published"}
    n = replay["n_by_split"].get(split, 0)
    total = replay["pnl_by_split"].get(split, 0.0)
    pub_n = pub.get("n_trades")
    pub_total = pub.get("total")
    n_ok = n == pub_n
    # Allow a tiny float drift; flag anything over a tenth of a cent.
    tot_ok = pub_total is not None and abs(total - pub_total) < 5e-4
    return {
        "trial_id": tid,
        "fee_basis": basis,
        "split": split,
        "replay_n": n,
        "published_n": pub_n,
        "replay_total": total,
        "published_total": pub_total,
        "n_match": n_ok,
        "total_match": tot_ok,
        "status": "ok" if (n_ok and tot_ok) else "DIFF",
    }


def WriteCsv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def MaybePlot(frame, path, title, ylabel):
    try:
        from Backtest.BankrollPlot import PlotBankroll
        PlotBankroll(frame, path=path, title=title, ylabel=ylabel)
        return True
    except Exception as exc:
        print(f"plot skipped ({path.name}): {exc}", flush=True)
        return False


def Main():
    t0 = time.time()
    jobs, meta, score = LoadJobs()
    print(f"jobs={len(jobs)} games={GAMES_PATH}", flush=True)
    workers = MaxWorkers()
    results = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=InitWorker) as pool:
        futs = {pool.submit(RunOne, job): job[-1] for job in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            tid, trades, n_games = fut.result()
            results[tid] = {"trades": trades, "n_games": n_games}
            print(f"  {i}/{len(jobs)} {tid} trades={len(trades)}", flush=True)

    checks = []
    tidy = []
    # Series name: family + venue + fee short tag, for the primary fee on all history.
    for tid, payload in results.items():
        row = meta[tid]
        for basis in FEE_BASES:
            replay = DailyEquity(payload["trades"], basis)
            for sp in ("tune", "holdout", "forward"):
                checks.append(Compare(tid, basis, sp, replay, row))
            if not replay["dates"]:
                continue
            fee_tag = "tim02" if basis == "tim_est_0.02" else "ktaker"
            series = f"{row['family']}|{row['venue']}|{fee_tag}|{tid}"
            iso_dates = [d.isoformat() for d in replay["dates"]]
            frame = FrameBankroll(iso_dates, replay["equity"], series=series, mode="auto")
            split_by_date = dict(zip(iso_dates, replay["splits"]))
            for item in TidyRows(frame):
                item["split"] = split_by_date.get(item["date"], "")
                item["venue"] = row["venue"]
                item["family"] = row["family"]
                item["trial_id"] = tid
                item["fee_basis"] = basis
                item["frame"] = "all_history"
                tidy.append(item)
            # Out-of-sample only (holdout+forward): restart equity at 0 on first OOS trade.
            oos = [tr for tr in payload["trades"] if tr["split"] in ("holdout", "forward")]
            oos_replay = DailyEquity(oos, basis)
            if oos_replay["dates"]:
                oos_series = series + "|oos"
                oos_dates = [d.isoformat() for d in oos_replay["dates"]]
                oos_frame = FrameBankroll(oos_dates, oos_replay["equity"], series=oos_series, mode="auto")
                oos_split = dict(zip(oos_dates, oos_replay["splits"]))
                for item in TidyRows(oos_frame):
                    item["split"] = oos_split.get(item["date"], "")
                    item["venue"] = row["venue"]
                    item["family"] = row["family"]
                    item["trial_id"] = tid
                    item["fee_basis"] = basis
                    item["frame"] = "holdout_forward"
                    tidy.append(item)

    fields = [
        "date", "series", "equity", "drawdown_pct", "season", "x", "x_kind",
        "split", "venue", "family", "trial_id", "fee_basis", "frame",
    ]
    csv_path = OUT_DIR / f"{BASE}.csv"
    WriteCsv(csv_path, tidy, fields)

    n_diff = sum(1 for c in checks if c["status"] == "DIFF")
    check_path = OUT_DIR / f"{BASE}_number_check.json"
    check_path.write_text(json.dumps({
        "games_path": str(GAMES_PATH),
        "n_jobs": len(jobs),
        "n_checks": len(checks),
        "n_diff": n_diff,
        "checks": checks,
        "elapsed_s": time.time() - t0,
    }, indent=2), encoding="utf-8")
    print(f"wrote {csv_path} rows={len(tidy)} diffs={n_diff}", flush=True)

    # Draft chart: Kalshi, Tim fee, OOS paths for the main families.
    want = set()
    for tid, row in meta.items():
        if row["venue"] == "kalshi" and row["exec"] == "taker":
            want.add(tid)
    plot_dates, plot_eq, plot_ser = [], [], []
    for tid, payload in results.items():
        if tid not in want:
            continue
        row = meta[tid]
        oos = [tr for tr in payload["trades"] if tr["split"] in ("holdout", "forward")]
        replay = DailyEquity(oos, "tim_est_0.02")
        if not replay["dates"]:
            continue
        label = row["family"]
        plot_dates.extend(d.isoformat() for d in replay["dates"])
        plot_eq.extend(replay["equity"])
        plot_ser.extend([label] * len(replay["dates"]))
    if plot_dates:
        frame = FrameBankroll(plot_dates, plot_eq, plot_ser, mode="auto")
        plotted = MaybePlot(
            frame,
            OUT_DIR / f"{BASE}_draft.png",
            title="Tune-selected in-play rules, holdout+forward (Kalshi, Tim $0.02 fee)",
            ylabel="Cumulative P&L, 1 contract per trade (US dollars)",
        )
        # Per-season drafts
        parts = SplitBySeason(plot_dates, plot_eq, plot_ser)
        for label, part in parts.items():
            MaybePlot(
                part,
                OUT_DIR / f"{BASE}_{label}_draft.png",
                title=f"Tune-selected in-play rules, {label} (Kalshi, Tim $0.02 fee)",
                ylabel="Cumulative P&L, 1 contract per trade (US dollars)",
            )
        print("plotted" if plotted else "csv only", frame["mode"], frame["seasons"], flush=True)

    print(f"elapsed {time.time()-t0:.1f}s", flush=True)
    return 0 if n_diff == 0 else 2


if __name__ == "__main__":
    sys.exit(Main())