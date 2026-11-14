"""Stage 2: Kalshi NBA player-prop pricing study and leak-free fee-adjusted backtest (2026-11-14).

Reads the panel from Props.BuildPricingPanel. Fee: $0.02 per $1 contract is Tim's working estimate
(fee_basis=tim_est_0.02), not a verified exchange fee. edge = p_model - (price + 0.02).
The study spec (grid, split, selection rule, verdict rule) is frozen and hashed BEFORE any holdout scoring.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import csv
import hashlib
import itertools
import json
import math
import pickle
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from Props.BuildPricingPanel import PANEL_PATH
from Backtest.ResidualSearch import HansenSpa

FEE = 0.02
FEE_BASIS = "tim_est_0.02"
OUT = Path(r"E:\AgentDocs\Basketbot\2026-11-14_prop-pricing")
BASE = "2026-11-14_prop-pricing"
REPS = 10000
SEED = 7

SPEC = {
    "title": "Kalshi NBA player-prop pricing study + fee-adjusted backtest",
    "fee": FEE, "fee_basis": FEE_BASIS, "fee_note": "Tim's working estimate, not a verified exchange fee",
    "edge": "yes: p - (yes_ask + fee); no: (1 - p) - (no_ask + fee), no_ask = 1 - yes_bid; one side per market",
    "model": "frozen NB/EB count models (Props.Model) fit on 2024-25 regular season only; features from earlier dates only",
    "tip": "scheduled tip from games_meta sched_ts (not Kalshi expected_expiration_time, which is often after tip)",
    "quote_rule": "last candle at or before tip - offset, at most 3600 s stale",
    "split": "chronological by priced date: first 60% of dates = tune (rule selection), last 40% = holdout",
    "grid": {
        "family": ["all", "3pm", "pts", "reb", "ast", "pra"],
        "side": ["best", "yes", "no"],
        "quote_offset_min": [120, 60, 15],
        "edge_cut": [0.03, 0.05, 0.10],
        "max_price": [0.60, 1.00],
    },
    "selection": "highest tune mean P&L per contract among configs with >= 200 tune bets and >= 20 tune days",
    "reference_rule": "R0 = family all, side best, T-60, edge >= 0.03, max_price 1.00 (the 2026-11-13 preregistered rule)",
    "metrics": "holdout mean P&L/contract with 95% day-block bootstrap CI (10,000 reps), CLV vs T-2 mid, Hansen SPA + White RC over all grid configs on holdout days (10,000 stationary day-block reps, mean block 5), MDE at 80% power = 2.8 x SE",
    "verdict_rule": "'prop edge exists' only if the selected config's holdout 95% CI lower bound > 0 AND holdout SPA p < 0.05; else 'no prop edge'",
    "daily_rule_rule": "freeze a daily rule only if the verdict is 'prop edge exists'; otherwise the daily check prints PASS",
    "no_bets": True,
}


def SpecHash(spec):
    d = dict(spec)
    d.pop("spec_hash", None)
    return hashlib.sha1(json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def Git():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def WriteCsv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def F(x, nd=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def C(x):  # cents
    return "NA" if x is None else f"{100 * x:+.1f}c"


def Log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def LL(p, y):
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def Q(r, off):
    return r["quotes"].get(off) or None


def TwoSided(q):
    return q is not None and q.get("ask") is not None and q.get("bid") is not None and q["bid"] > 0 and q["ask"] < 1 and q["ask"] > q["bid"]


def Mid(q):
    return 0.5 * (q["ask"] + q["bid"]) if TwoSided(q) else None


def DayBoot(vals, days, reps=REPS, seed=SEED, level=0.95):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "n_days": 0, "se": None}
    keys = sorted(set(days))
    ix = {k: i for i, k in enumerate(keys)}
    sums = np.zeros(len(keys)); cnts = np.zeros(len(keys))
    for v, d in zip(vals, days):
        sums[ix[d]] += v; cnts[ix[d]] += 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(reps, len(keys)))
    boot = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
    a = (1 - level) / 2
    lo, hi = np.quantile(boot, [a, 1 - a])
    return {"mean": float(vals.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(vals)),
            "n_days": len(keys), "se": float(boot.std(ddof=1))}


def Decide(r, off, side, cut, maxp):
    q = Q(r, off)
    if q is None or q.get("ask") is None:
        return None
    p = r["model_p"]
    ya = q["ask"]
    na = None if q.get("bid") is None else 1.0 - q["bid"]
    cands = []
    if side in ("best", "yes") and ya is not None and 0 < ya <= maxp:
        cands.append(("yes", p - (ya + FEE), ya))
    if side in ("best", "no") and na is not None and 0 < na <= maxp and na < 1:
        cands.append(("no", (1 - p) - (na + FEE), na))
    if not cands:
        return None
    s, e, fill = max(cands, key=lambda c: c[1])
    if e < cut:
        return None
    win = r["yes"] if s == "yes" else 1 - r["yes"]
    pnl = win - (fill + FEE)
    cq = Q(r, "T2")
    cm = Mid(cq)
    clv = None if cm is None else ((cm if s == "yes" else 1 - cm) - fill)
    return {"side": s, "edge": e, "fill": fill, "win": win, "pnl": pnl, "clv": clv}


def Tier(pm):
    if pm is None:
        return "NA"
    pm = float(pm)
    return "<20" if pm < 20 else ("20-28" if pm < 28 else ("28-34" if pm < 34 else "34+"))


def Main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = dict(SPEC)
    spec["frozen_at_et"] = datetime.now().astimezone().isoformat(timespec="seconds")
    spec_path = OUT / f"{BASE}_spec.json"
    if spec_path.exists():
        old = json.loads(spec_path.read_text(encoding="utf-8"))
        if SpecHash({k: v for k, v in old.items() if k != "frozen_at_et"}) != SpecHash(SPEC):
            raise SystemExit("frozen spec changed; refuse to rescore")
        spec = old
    else:
        spec["spec_hash"] = SpecHash(SPEC)
        spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    spec_hash = SpecHash(SPEC)
    Log(f"spec hash {spec_hash}")

    blob = pickle.loads(PANEL_PATH.read_bytes())
    rows = blob["rows"]
    Log(f"panel rows={len(rows)} skipped={blob['skipped']}")
    tidy = []  # long-format chart-ready rows

    def T(section, gtype, group, metric, value, n=None, split=None):
        tidy.append({"section": section, "group_type": gtype, "group": group, "metric": metric,
                     "value": value, "n": n, "split": split})

    # ---------- coverage ----------
    cov = []
    for s in sorted({r["series"] for r in rows}):
        rs = [r for r in rows if r["series"] == s]
        q60 = [r for r in rs if Q(r, "T60") and Q(r, "T60").get("ask") is not None]
        cov.append({"series": s, "n_panel": len(rs), "n_quoted_T60": len(q60),
                    "n_two_sided_T60": sum(1 for r in q60 if TwoSided(Q(r, "T60"))),
                    "n_dnp_carry_forward": sum(1 for r in rs if str(r["match"]).startswith("carry")),
                    "date_min": min(r["date"] for r in rs), "date_max": max(r["date"] for r in rs),
                    "n_days": len({r["date"] for r in rs})})
    for c in cov:
        for k in ("n_panel", "n_quoted_T60", "n_two_sided_T60"):
            T("coverage", "series", c["series"], k, c[k])
    emis = [r["exp_minus_sched_min"] for r in rows if r.get("exp_minus_sched_min") is not None]
    occ = [r["occ_minus_sched_min"] for r in rows if r.get("occ_minus_sched_min") is not None]
    tipcheck = {
        "n_exp": len(emis), "exp_minus_sched_median_min": float(np.median(emis)) if emis else None,
        "share_exp_gt_60min_after_sched": float(np.mean(np.array(emis) > 60)) if emis else None,
        "n_occ": len(occ), "occ_minus_sched_median_min": float(np.median(occ)) if occ else None,
        "share_occ_after_sched": float(np.mean(np.array(occ) > 0)) if occ else None,
    }

    # ---------- Q1 pricing structure at T-60 ----------
    struct = []
    for fam in ["3pm", "pts", "reb", "ast", "pra"]:
        for prot in (False, True):
            rs = [r for r in rows if r["family"] == fam and r["protected"] == prot and Q(r, "T60") and Q(r, "T60").get("ask") is not None]
            if not rs:
                continue
            two = [r for r in rs if TwoSided(Q(r, "T60"))]
            sp = [Q(r, "T60")["ask"] - Q(r, "T60")["bid"] for r in two]
            struct.append({"family": fam, "protected": prot, "n_quoted": len(rs),
                           "share_two_sided": len(two) / len(rs),
                           "mean_spread_two_sided": float(np.mean(sp)) if sp else None,
                           "median_spread_two_sided": float(np.median(sp)) if sp else None,
                           "mean_hold_all": float(np.mean([Q(r, "T60")["ask"] + (1 - (Q(r, "T60").get("bid") or 0)) - 1 for r in rs])),
                           "mean_yes_ask": float(np.mean([Q(r, "T60")["ask"] for r in rs])),
                           "mean_volume": float(np.mean([r["volume"] for r in rs])),
                           "strikes": " ".join(str(k) for k in sorted({r["k"] for r in rs})[:12])})
    for s in struct:
        g = f"{s['family']}{'_prot' if s['protected'] else ''}"
        for k in ("share_two_sided", "mean_spread_two_sided", "mean_hold_all", "mean_volume"):
            T("q1_structure", "family", g, k, s[k], s["n_quoted"])
    # ladders
    lad = defaultdict(list)
    for r in rows:
        q = Q(r, "T60")
        if q and q.get("ask") is not None:
            lad[(r["event_ticker"], r["player_name"], r["series"])].append((r["k"], q))
    n_pairs = n_viol = n_arb = n_arb_fee = 0
    yes_sum_over = []
    for key, lst in lad.items():
        lst.sort(key=lambda x: x[0])
        for (k1, q1), (k2, q2) in zip(lst, lst[1:]):
            n_pairs += 1
            if q2["ask"] > q1["ask"] + 1e-9:
                n_viol += 1
            b2 = q2.get("bid") or 0
            if b2 > q1["ask"]:
                n_arb += 1
            if b2 > q1["ask"] + 2 * FEE:
                n_arb_fee += 1
    ladder = {"n_ladders": len(lad), "n_adjacent_pairs": n_pairs,
              "share_ask_monotonicity_violations": n_viol / n_pairs if n_pairs else None,
              "n_cross_strike_arbs_prefee": n_arb, "n_cross_strike_arbs_after_tim_fee": n_arb_fee}
    # movement first -> T60 -> T2
    mov = []
    for fam in ["all", "3pm", "pts", "reb", "ast", "pra"]:
        rs = [r for r in rows if (fam == "all" or r["family"] == fam)]
        trip = [r for r in rs if Mid(Q(r, "first")) is not None and Mid(Q(r, "T60")) is not None and Mid(Q(r, "T2")) is not None]
        if not trip:
            continue
        m0 = np.array([Mid(Q(r, "first")) for r in trip]); m60 = np.array([Mid(Q(r, "T60")) for r in trip])
        m2 = np.array([Mid(Q(r, "T2")) for r in trip]); y = np.array([r["yes"] for r in trip])
        lead = np.array([(Q(r, "T60")["t"] - Q(r, "first")["t"]) / 60 for r in trip])
        d_late = m2 - m60
        resid = y - m60
        mov.append({"family": fam, "n": len(trip), "median_first_candle_lead_min_before_T60": float(np.median(lead)),
                    "mean_abs_move_first_to_T60": float(np.mean(np.abs(m60 - m0))),
                    "mean_abs_move_last_hour": float(np.mean(np.abs(d_late))),
                    "share_moved_last_hour": float(np.mean(np.abs(d_late) > 1e-9)),
                    "corr_last_hour_move_vs_resid": float(np.corrcoef(d_late, resid)[0, 1]) if d_late.std() > 0 else None,
                    "brier_first": float(np.mean((m0 - y) ** 2)), "brier_T60": float(np.mean((m60 - y) ** 2)),
                    "brier_T2": float(np.mean((m2 - y) ** 2))})
    for m in mov:
        for k in ("mean_abs_move_first_to_T60", "mean_abs_move_last_hour", "corr_last_hour_move_vs_resid", "brier_first", "brier_T60", "brier_T2"):
            T("q1_movement", "family", m["family"], k, m[k], m["n"])

    # ---------- Q2 calibration + disagreement ----------
    two60 = [r for r in rows if TwoSided(Q(r, "T60"))]
    rel = []
    edges = np.linspace(0, 1, 11)
    for src in ("kalshi_mid_T60", "model"):
        vals = np.array([Mid(Q(r, "T60")) if src != "model" else r["model_p"] for r in two60])
        y = np.array([r["yes"] for r in two60])
        b = np.clip(np.digitize(vals, edges) - 1, 0, 9)
        for i in range(10):
            mk = b == i
            if mk.sum():
                rel.append({"source": src, "bin_lo": edges[i], "bin_hi": edges[i + 1], "n": int(mk.sum()),
                            "mean_pred": float(vals[mk].mean()), "mean_outcome": float(y[mk].mean())})
                T("q2_reliability", src, f"{edges[i]:.1f}-{edges[i+1]:.1f}", "mean_outcome", float(y[mk].mean()), int(mk.sum()))
                T("q2_reliability", src, f"{edges[i]:.1f}-{edges[i+1]:.1f}", "mean_pred", float(vals[mk].mean()), int(mk.sum()))
    calib = []
    for fam in ["all", "3pm", "pts", "reb", "ast", "pra"]:
        rs = [r for r in two60 if fam == "all" or r["family"] == fam]
        if len(rs) < 30:
            continue
        y = np.array([r["yes"] for r in rs]); mid = np.array([Mid(Q(r, "T60")) for r in rs]); pm = np.array([r["model_p"] for r in rs])
        bits = [(LL(mm, yy) - LL(pp, yy)) / math.log(2) for mm, pp, yy in zip(mid, pm, y)]
        gap = DayBoot(bits, [r["date"] for r in rs], level=0.90)
        d = pm - mid; res = y - mid
        slope = float(np.dot(d - d.mean(), res - res.mean()) / max(np.dot(d - d.mean(), d - d.mean()), 1e-12))
        calib.append({"family": fam, "n": len(rs), "brier_mid": float(np.mean((mid - y) ** 2)), "brier_model": float(np.mean((pm - y) ** 2)),
                      "ll_mid": float(np.mean([LL(a, b) for a, b in zip(mid, y)])), "ll_model": float(np.mean([LL(a, b) for a, b in zip(pm, y)])),
                      "gap_bits_mid_minus_model": gap["mean"], "gap_lo90": gap["lo"], "gap_hi90": gap["hi"],
                      "mean_model_minus_mid": float(d.mean()), "mean_abs_disagreement": float(np.abs(d).mean()),
                      "slope_resid_on_disagreement": slope, "base_rate": float(y.mean())})
    for c in calib:
        for k in ("brier_mid", "brier_model", "ll_mid", "ll_model", "gap_bits_mid_minus_model", "mean_abs_disagreement", "slope_resid_on_disagreement"):
            T("q2_calibration", "family", c["family"], k, c[k], c["n"])
    dis = []
    for gtype, keyf in (("family", lambda r: r["family"]), ("strike_bucket", lambda r: f"{r['family']}:{r['strike_bucket']}"),
                        ("tier_proj_min", lambda r: Tier(r.get("proj_min"))), ("protected", lambda r: str(r["protected"]))):
        grp = defaultdict(list)
        for r in two60:
            grp[keyf(r)].append(r)
        for g, rs in sorted(grp.items()):
            if len(rs) < 30:
                continue
            d = np.array([r["model_p"] - Mid(Q(r, "T60")) for r in rs]); res = np.array([r["yes"] - Mid(Q(r, "T60")) for r in rs])
            slope = float(np.dot(d - d.mean(), res - res.mean()) / max(np.dot(d - d.mean(), d - d.mean()), 1e-12))
            dis.append({"group_type": gtype, "group": g, "n": len(rs), "mean_model_minus_mid": float(d.mean()),
                        "mean_abs_disagreement": float(np.abs(d).mean()), "mean_outcome_minus_mid": float(res.mean()),
                        "slope_resid_on_disagreement": slope})
            T("q2_disagreement", gtype, g, "mean_model_minus_mid", float(d.mean()), len(rs))
            T("q2_disagreement", gtype, g, "slope_resid_on_disagreement", slope, len(rs))
    for off in ("first", "T240", "T120", "T60", "T30", "T15", "T2"):
        rs = [r for r in rows if TwoSided(Q(r, off))]
        if len(rs) < 30:
            continue
        d = np.array([r["model_p"] - Mid(Q(r, off)) for r in rs]); y = np.array([r["yes"] for r in rs]); mid = np.array([Mid(Q(r, off)) for r in rs])
        dis.append({"group_type": "time_before_tip", "group": off, "n": len(rs), "mean_model_minus_mid": float(d.mean()),
                    "mean_abs_disagreement": float(np.abs(d).mean()), "brier_mid": float(np.mean((mid - y) ** 2)),
                    "brier_model": float(np.mean((np.array([r["model_p"] for r in rs]) - y) ** 2))})
        T("q2_disagreement", "time_before_tip", off, "mean_abs_disagreement", float(np.abs(d).mean()), len(rs))

    # ---------- Q3 backtest ----------
    priced_dates = sorted({r["date"] for r in rows if Q(r, "T60") and Q(r, "T60").get("ask") is not None})
    cut_i = int(round(0.6 * len(priced_dates)))
    tune_dates = set(priced_dates[:cut_i]); hold_dates = set(priced_dates[cut_i:])
    split_info = {"n_priced_dates": len(priced_dates), "tune_first": priced_dates[0], "tune_last": priced_dates[cut_i - 1],
                  "hold_first": priced_dates[cut_i], "hold_last": priced_dates[-1], "n_tune_dates": len(tune_dates), "n_hold_dates": len(hold_dates)}
    Log(f"split {split_info}")
    g = SPEC["grid"]
    configs = list(itertools.product(g["family"], g["side"], g["quote_offset_min"], g["edge_cut"], g["max_price"]))
    hold_list = sorted(hold_dates); hix = {d: i for i, d in enumerate(hold_list)}
    all_list = priced_dates; aix = {d: i for i, d in enumerate(all_list)}
    Dh = np.zeros((len(hold_list), len(configs)))
    Da = np.zeros((len(all_list), len(configs)))
    registry, bets_by_cfg = [], {}
    for j, (fam, side, off, cut, maxp) in enumerate(configs):
        cid = f"{fam}|{side}|T{off}|e{cut:.2f}|max{maxp:.2f}"
        bets = []
        for r in rows:
            if fam != "all" and r["family"] != fam:
                continue
            dcs = Decide(r, f"T{off}", side, cut, maxp)
            if dcs:
                dcs.update({"date": r["date"], "ticker": r["ticker"], "series": r["series"], "family": r["family"],
                            "k": r["k"], "player_name": r["player_name"], "model_p": r["model_p"], "match": r["match"],
                            "protected": r["protected"]})
                bets.append(dcs)
        bets_by_cfg[cid] = bets
        tb = [b for b in bets if b["date"] in tune_dates]; hb = [b for b in bets if b["date"] in hold_dates]
        for D, idx, bl in ((Dh, hix, hb), (Da, aix, bets)):
            s = defaultdict(float); n = defaultdict(int)
            for b in bl:
                s[b["date"]] += b["pnl"]; n[b["date"]] += 1
            for d in s:
                D[idx[d], j] = s[d] / n[d]
        registry.append({"config_id": cid, "family": fam, "side": side, "offset_min": off, "edge_cut": cut, "max_price": maxp,
                         "tune_n": len(tb), "tune_days": len({b["date"] for b in tb}),
                         "tune_mean_pnl": float(np.mean([b["pnl"] for b in tb])) if tb else None,
                         "hold_n": len(hb), "hold_mean_pnl": float(np.mean([b["pnl"] for b in hb])) if hb else None,
                         "all_n": len(bets), "all_mean_pnl": float(np.mean([b["pnl"] for b in bets])) if bets else None})
    eligible = [x for x in registry if x["tune_n"] >= 200 and x["tune_days"] >= 20 and x["tune_mean_pnl"] is not None]
    sel = max(eligible, key=lambda x: x["tune_mean_pnl"]) if eligible else None
    # Freeze the selection BEFORE holdout metrics are computed for it.
    sel_spec = {"spec_hash": spec_hash, "selected_config": None if sel is None else {k: sel[k] for k in ("config_id", "family", "side", "offset_min", "edge_cut", "max_price", "tune_n", "tune_mean_pnl")},
                "selected_at_et": datetime.now().astimezone().isoformat(timespec="seconds"), "split": split_info}
    sel_spec["selection_hash"] = hashlib.sha1(json.dumps(sel_spec, sort_keys=True).encode()).hexdigest()
    (OUT / f"{BASE}_selection.json").write_text(json.dumps(sel_spec, indent=1), encoding="utf-8")
    Log(f"selected {sel_spec['selected_config']}")

    def Eval(cid, dates):
        bl = [b for b in bets_by_cfg[cid] if b["date"] in dates]
        res = DayBoot([b["pnl"] for b in bl], [b["date"] for b in bl])
        clv = [b["clv"] for b in bl if b["clv"] is not None]
        res["clv_mean"] = float(np.mean(clv)) if clv else None
        res["clv_n"] = len(clv)
        res["wins"] = int(sum(b["win"] for b in bl))
        res["mean_claimed_edge"] = float(np.mean([b["edge"] for b in bl])) if bl else None
        res["mde80"] = None if res["se"] is None else 2.8 * res["se"]
        return res, bl

    r0 = "all|best|T60|e0.03|max1.00"
    out_rules = {}
    for name, cid in (("selected", sel["config_id"] if sel else None), ("R0_prereg_rule", r0)):
        if cid is None:
            continue
        hold, hb = Eval(cid, hold_dates)
        tune, _ = Eval(cid, tune_dates)
        full, _ = Eval(cid, set(priced_dates))
        out_rules[name] = {"config_id": cid, "holdout": hold, "tune": tune, "full_oos_window": full}
        if name == "selected":
            sel_bets = hb
        for sp, res in (("tune", tune), ("holdout", hold), ("full", full)):
            T("q3_backtest", "rule", name, "mean_pnl_per_contract", res["mean"], res["n"], sp)
            T("q3_backtest", "rule", name, "ci95_lo", res["lo"], res["n"], sp)
            T("q3_backtest", "rule", name, "ci95_hi", res["hi"], res["n"], sp)
            T("q3_backtest", "rule", name, "clv_mean", res["clv_mean"], res["clv_n"], sp)
    Log("SPA holdout...")
    spa_h = HansenSpa(Dh, reps=REPS, seed=SEED, mean_block=5)
    Log("SPA full...")
    spa_a = HansenSpa(Da, reps=REPS, seed=SEED, mean_block=5)
    spa_h["best_config"] = configs[spa_h["best_k"]]; spa_a["best_config"] = configs[spa_a["best_k"]]
    # per-family breakdown of R0 on holdout
    fam_rows = []
    for fam in ["3pm", "pts", "reb", "ast", "pra"]:
        for prot in (False, True):
            bl = [b for b in bets_by_cfg[r0] if b["family"] == fam and b["protected"] == prot]
            for sp, ds in (("tune", tune_dates), ("holdout", hold_dates)):
                x = [b for b in bl if b["date"] in ds]
                if not x:
                    continue
                res = DayBoot([b["pnl"] for b in x], [b["date"] for b in x])
                fam_rows.append({"rule": "R0", "family": fam, "protected": prot, "split": sp, "n": res["n"], "mean_pnl": res["mean"], "lo95": res["lo"], "hi95": res["hi"],
                                 "yes_share": float(np.mean([b["side"] == "yes" for b in x])), "dnp_carry_share": float(np.mean([str(b["match"]).startswith("carry") for b in x]))})
                T("q3_by_family", "family", f"{fam}{'_prot' if prot else ''}", "mean_pnl_per_contract", res["mean"], res["n"], sp)

    hs = out_rules.get("selected", {}).get("holdout") or {}
    exists = bool(hs.get("lo") is not None and hs["lo"] > 0 and spa_h["spa_p"] < 0.05)
    verdict = "prop edge exists" if exists else "no prop edge"
    vline = (f"VERDICT: {verdict} | holdout edge {C(hs.get('mean'))}/contract after Tim's $0.02 fee estimate, "
             f"95% CI [{C(hs.get('lo'))}, {C(hs.get('hi'))}], N={hs.get('n')} bets on {hs.get('n_days')} days, "
             f"SPA p={F(spa_h['spa_p'], 3)} over {len(configs)} configs (selected: {out_rules.get('selected', {}).get('config_id')})")

    # ---------- Q4 daily rule ----------
    rule = {"name": "nba_prop_daily_rule", "created_at_et": datetime.now().astimezone().isoformat(timespec="seconds"),
            "study_spec_hash": spec_hash, "fee": FEE, "fee_basis": FEE_BASIS, "min_edge": 0.03,
            "status": "qualified" if exists else "none_qualifies",
            "reason": "holdout CI lower bound > 0 and SPA p < 0.05" if exists else
                      "no config cleared the frozen verdict rule (holdout CI lower bound > 0 AND SPA p < 0.05); daily check must print PASS",
            "rule": None if not exists else {k: sel[k] for k in ("family", "side", "offset_min", "edge_cut", "max_price")},
            "monitor_only_candidate": None if sel is None else {k: sel[k] for k in ("config_id", "family", "side", "offset_min", "edge_cut", "max_price")},
            "moneyline_status": "none_qualifies",
            "moneyline_reason": "2026-11-13 market-blend (3 qualifying bets, w_m CI spans 0) and residual-search (SPA p=0.82, 3c+ edge CI [-6.0c, +13.6c]) both PASS"}
    rule["rule_hash"] = hashlib.sha1(json.dumps({k: v for k, v in rule.items() if k != "created_at_et"}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (OUT / f"{BASE}_rule.json").write_text(json.dumps(rule, indent=1), encoding="utf-8")

    summary = {"verdict": verdict, "verdict_line": vline, "spec_hash": spec_hash, "selection_hash": sel_spec["selection_hash"],
               "rule_hash": rule["rule_hash"], "rule": rule, "git": Git(), "fee_basis": FEE_BASIS,
               "split": split_info, "n_configs": len(configs), "spa_holdout": spa_h, "spa_full_window": spa_a,
               "rules": out_rules, "coverage": cov, "panel_skipped": blob["skipped"], "tip_check": tipcheck,
               "ladder": ladder, "structure": struct, "movement": mov, "calibration": calib,
               "generated_at_et": datetime.now().astimezone().isoformat(timespec="seconds"), "elapsed_s": time.time() - t0}
    (OUT / f"{BASE}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    WriteCsv(OUT / f"{BASE}.csv", tidy)
    WriteCsv(OUT / f"{BASE}_trials.csv", registry)
    WriteCsv(OUT / f"{BASE}_reliability.csv", rel)
    WriteCsv(OUT / f"{BASE}_disagreement.csv", dis)
    WriteCsv(OUT / f"{BASE}_by-family.csv", fam_rows)
    WriteCsv(OUT / f"{BASE}_structure.csv", struct)
    WriteCsv(OUT / f"{BASE}_coverage.csv", cov)
    WriteCsv(OUT / f"{BASE}_holdout-bets.csv", [{k: v for k, v in b.items()} for b in (sel_bets if sel else [])])
    (OUT / "_verdict_line.txt").write_text(vline + "\n", encoding="utf-8")
    Log(vline)
    Log(f"DONE elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    Main()
