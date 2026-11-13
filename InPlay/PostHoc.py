r"""Post-hoc follow-up on the favorite-longshot pattern seen in calibration.

These trials were defined AFTER the holdout calibration was seen, so their holdout numbers are contaminated.
Only the tune split is a fair selection basis and the forward split from here on is the real test.

    .venv\Scripts\python.exe -m InPlay.PostHoc --workers 8
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

from InPlay.Grid import TrialId
from InPlay.Research import BASES, AppendMd, AppendTrials, CsvRows, Metrics, Now, SaveRes, _Init, _Run, _Short


def Variants():
    v = []
    for mins in (48, 24, 12, 6):
        for lo, hi in ((0.90, 0.95), (0.95, 0.98), (0.90, 0.97)):
            v.append(("late_fav", dict(mins=mins, lo=lo, hi=hi)))
    return v


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    jobs = []
    for fam, P in Variants():
        jobs.append((fam, P, "kalshi", "taker", ("tune", "holdout", "forward")))
        jobs.append((fam, P, "kalshi", "maker", ("tune", "holdout", "forward")))
        jobs.append((fam, P, "pm", "taker", ("tune", "holdout", "forward")))
    md = ["## POST-HOC (holdout-contaminated): buy the current favorite at 0.90-0.98 and hold to settlement", "",
          "Defined after seeing that 0.90-0.99 favorites win ~1-2 points more often than priced. Treat holdout as in-sample;",
          "only future forward games can confirm. Fee basis kalshi_taker (maker fills use the maker fee); Tim est. in CSV.", "",
          "| venue | exec | rule | tune n / P&L | holdout n / P&L [CI] | forward n / P&L [CI] |", "|---|---|---|---|---|---|"]
    rows, trials, out = [], [], {}
    with ProcessPoolExecutor(max_workers=min(args.workers, 10), initializer=_Init) as ex:
        for (fam, P, venue, xm, _), trades, gsp, dsp in ex.map(_Run, jobs):
            tid = TrialId(fam + "_posthoc", P, venue, xm)
            res = {}
            for sp in ("tune", "holdout", "forward"):
                tr = [t for t in trades if t["split"] == sp]
                res[sp] = {b: Metrics(tr, gsp.get(sp, 0), dsp.get(sp, []), b) for b in BASES}
                for b in BASES:
                    m = res[sp][b]
                    rows.append(dict(section="posthoc_backtest", venue=venue, split=sp, family="late_fav_posthoc", trial_id=tid,
                                     fee_basis=b, metric="pnl_per_trade", value=m["pnl_per_trade"], lo=m["lo"], hi=m["hi"],
                                     n=m["n_trades"], label=_Short(P) + "|" + xm))
            k = {sp: res[sp]["kalshi_taker"] for sp in res}
            cell = lambda m: f"{m['n_trades']} / {m['pnl_per_trade']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}]" if m["n_trades"] else "0"
            md.append(f"| {venue} | {xm} | {_Short(P)} | {cell(k['tune'])} | {cell(k['holdout'])} | {cell(k['forward'])} |")
            trials.append(dict(trial_id=tid, created_at=Now(), family="late_fav_posthoc", params=P, venue=venue, exec=xm,
                               tag="post_hoc_after_holdout_calibration_seen", split_scored="tune+holdout+forward", res=res))
            out[tid] = dict(params=P, venue=venue, exec=xm, res=res)
    SaveRes("posthoc", out)
    CsvRows(rows)
    AppendTrials(trials)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())