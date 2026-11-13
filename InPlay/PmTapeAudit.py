r"""Polymarket timestamp audit: do the prices-history labels run late? (caveat 1 / Round 6A)

For a seeded sample of Polymarket games per split, pull public on-chain trades (data-api v2 /trades, both outcome tokens,
start+30 min to start+150 min), turn them into a home-price "last trade at or before t" series, and find the offset L
(seconds) that best aligns the stored 1-minute prices-history labels with the trades: history(T) ~ trades(T + L).
L = -60 means the history label is a minute late (assumption H2, used in the final run); L = 0 means it is on time.

    .venv\Scripts\python.exe -m InPlay.PmTapeAudit
"""

import bisect
import json
import os
import random
import sys

import numpy as np

from InPlay.Align import SeasonOf
from InPlay.Http import GetJson, HttpError, ReadJson, WriteJsonAtomic
from InPlay.Load import SplitOf
from InPlay.Research import AppendMd, CsvRows

ROOT = os.path.join("data", "inplay", "polymarket")
OUT = os.path.join(ROOT, "trades")
OFFS = list(range(-150, 151, 15))


def Cid(row):
    p = os.path.join(OUT, row["slug"] + ".cid.json")
    if os.path.exists(p):
        return ReadJson(p)["cid"]
    m = GetJson("https://gamma-api.polymarket.com/markets/" + str(row["id"]))
    WriteJsonAtomic(p, {"cid": m["conditionId"]})
    return m["conditionId"]


def Pull(row):
    p = os.path.join(OUT, row["slug"] + ".json")
    if os.path.exists(p):
        return ReadJson(p)["trades"]
    cid = Cid(row)
    s, e = row["start_ts"] + 1800, row["start_ts"] + 9000
    rows, cursor, pages = [], None, 0
    while pages < 120:
        params = {"condition": cid, "start": s, "end": e, "limit": 500}
        if cursor:
            params["cursor"] = cursor
        r = GetJson("https://data-api.polymarket.com/v2/trades", params=params)
        pages += 1
        for t in r.get("data") or []:
            px = float(t["price"])
            rows.append([int(t["timestamp"]), px if t["token_id"] == row["home_token"] else 1 - px])
        cursor = (r.get("pagination") or {}).get("next_cursor")
        if not cursor:
            break
    rows.sort()
    WriteJsonAtomic(p, {"slug": row["slug"], "trades": rows, "pages": pages})
    return rows


def BestOffset(hist, trades, s, e):
    ts = [t for t, _ in trades]
    px = [p for _, p in trades]
    H = [(h["t"], h["p"]) for h in hist if s + 120 <= h["t"] <= e - 180]
    if len(H) < 60 or len(ts) < 200:
        return None
    hv = np.array([p for _, p in H])
    dh = np.diff(hv)
    out = {}
    for L in OFFS:
        tv = []
        for T, _ in H:
            j = bisect.bisect_right(ts, T + L) - 1
            tv.append(px[j] if j >= 0 else np.nan)
        dt = np.diff(np.array(tv))
        ok = ~np.isnan(dt)
        if ok.sum() > 50 and dh[ok].std() > 0 and dt[ok].std() > 0:
            out[L] = float(np.corrcoef(dh[ok], dt[ok])[0, 1])
    return out


def Main():
    rows = ReadJson(os.path.join(ROOT, "markets.json"))
    by = {}
    for r in rows:
        if not r.get("start_ts") or not os.path.exists(os.path.join(ROOT, "prices", r["slug"] + ".json")):
            continue
        sp = SplitOf(SeasonOf(r["start_ts"]))
        by.setdefault(sp, []).append(r)
    md = ["## Polymarket timestamp audit: prices-history labels vs on-chain trade timestamps", "",
          "Correlation of 1-minute changes: stored history at label T vs last on-chain trade at T + L (seconds), averaged over games. "
          "The best L tells where the label sits: L = -60 means labels run a minute late (the final run's assumption), L = 0 on time.", "",
          "| split | games | best L (s) | corr at L=-60 | corr at L=0 | corr at L=+60 | games whose best L <= -45 |", "|---|---|---|---|---|---|---|"]
    out_rows = []
    for sp in ("tune", "holdout", "forward"):
        pool = by.get(sp, [])
        random.Random(31).shuffle(pool)
        curves = []
        for r in pool[:15]:
            try:
                tr = Pull(r)
            except HttpError as e:
                print("ERR", r["slug"], e, flush=True)
                continue
            hist = ReadJson(os.path.join(ROOT, "prices", r["slug"] + ".json"))["history"]
            c = BestOffset(hist, tr, r["start_ts"] + 1800, r["start_ts"] + 9000)
            if c:
                curves.append(c)
            print(sp, r["slug"], len(tr), max(c, key=c.get) if c else None, flush=True)
        if not curves:
            continue
        mean = {L: float(np.mean([c[L] for c in curves if L in c])) for L in OFFS}
        best = max(mean, key=mean.get)
        late = sum(1 for c in curves if max(c, key=c.get) <= -45)
        md.append(f"| {sp} | {len(curves)} | {best:+d} | {mean[-60]:.3f} | {mean[0]:.3f} | {mean[60]:.3f} | {late}/{len(curves)} |")
        for L, v in mean.items():
            out_rows.append(dict(section="pm_timestamp_audit", venue="pm", split=sp, family="describe", metric="corr_hist_vs_trades",
                                 value=v, n=len(curves), label=f"L={L:+d}s"))
    md.append("")
    CsvRows(out_rows)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())