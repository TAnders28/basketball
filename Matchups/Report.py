r"""
Score a finished Matchups.RunAll run. Calibration/stacking weights are fit on the TUNE season only and frozen.

    .venv\Scripts\python.exe -m Matchups.Report --out-dir E:\AgentDocs\Basketbot\2026-11-12_matchup-models
"""

import argparse
import csv
import json
import os

import numpy as np

from Matchups import Scoring
from Matchups.RunAll import FORWARD, HOLDOUT, RUN_DIR, TUNE

BASE = "2026-11-12_matchup-models"
SPLITS = {TUNE: "tune_2024-25", HOLDOUT: "holdout_2025-26", FORWARD: "forward_2026-27_to_nov11"}


def Features(rows, variant):
    L = Scoring.Logit
    elo = np.array([r["elo"] for r in rows])
    out = {"elo_predictday_v1": None,          # raw, no fitting
           "elo_recalibrated": L(elo)[:, None],
           f"m1_rapm_{variant}": np.array([r[f"rapm_margin_{variant}"] for r in rows])[:, None],
           f"m2_matchup_{variant}": np.array([r[f"mm_margin_{variant}"] for r in rows])[:, None],
           f"m3_sim_raw_{variant}": None,
           f"m3_sim_cal_{variant}": L([r[f"sim_p_{variant}"] for r in rows])[:, None],
           f"m2b_matchup_team_{variant}": np.array([r[f"mmt_margin_{variant}"] for r in rows])[:, None],
           f"stack_elo_m1_{variant}": np.column_stack([L(elo), [r[f"rapm_margin_{variant}"] for r in rows]]),
           f"stack_elo_m2_m2b_{variant}": np.column_stack([L(elo), [r[f"mm_margin_{variant}"] for r in rows],
                                                           [r[f"mmt_margin_{variant}"] for r in rows]]),
           f"stack_elo_m1_m2_m2b_{variant}": np.column_stack([L(elo), [r[f"rapm_margin_{variant}"] for r in rows],
                                                              [r[f"mm_margin_{variant}"] for r in rows],
                                                              [r[f"mmt_margin_{variant}"] for r in rows]])}
    raw = {"elo_predictday_v1": elo, f"m3_sim_raw_{variant}": np.array([r[f"sim_p_{variant}"] for r in rows])}
    return out, raw


def Complete(r, variant):
    keys = ["elo", f"rapm_margin_{variant}", f"mm_margin_{variant}", f"mmt_margin_{variant}", f"sim_p_{variant}"]
    return all(r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k])) for k in keys)


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=os.path.join(RUN_DIR, "run_2026-11-12.json"))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    run = json.load(open(args.run, encoding="utf-8"))
    os.makedirs(args.out_dir, exist_ok=True)
    metrics_rows, detail = [], {"calibration": {}, "weights": {}, "assignment": {}, "pairs": {}, "margin_total": {}}
    for variant in ("pregame", "oracle"):
        rows = [r for r in run["rows"] if Complete(r, variant) and r["season"] in SPLITS]
        by_season = {s: [r for r in rows if r["season"] == s] for s in SPLITS}
        feats = {s: Features(by_season[s], variant) for s in SPLITS}
        y = {s: np.array([r["home_win"] for r in by_season[s]], dtype=float) for s in SPLITS}
        dates = {s: [r["game_date"] for r in by_season[s]] for s in SPLITS}
        names = list(feats[TUNE][0].keys())
        if variant == "oracle":
            names = [n for n in names if n.endswith("_oracle")]
        weights = {}
        for n in names:
            X = feats[TUNE][0][n]
            if X is not None:
                weights[n] = Scoring.FitLogistic(X, y[TUNE])
                detail["weights"][n] = [round(float(w), 5) for w in weights[n]]
        for s, split in SPLITS.items():
            preds = {}
            for n in names:
                X = feats[s][0][n]
                preds[n] = feats[s][1][n] if X is None else Scoring.PredictLogistic(weights[n], X)
            base = feats[s][1]["elo_predictday_v1"]
            recal = Scoring.PredictLogistic(weights["elo_recalibrated"], feats[s][0]["elo_recalibrated"]) if "elo_recalibrated" in weights else None
            for n in names:
                m = Scoring.Metrics(preds[n], y[s])
                b = Scoring.BootstrapDiff(preds[n], base, y[s], dates[s])
                row = {"date": "2026-11-12", "series": n, "split": split, "season": s, "n": m["n"],
                       "log_loss": round(m["log_loss"], 5), "brier": round(m["brier"], 5),
                       "accuracy": round(m["accuracy"], 4), "ece": m["ece"],
                       "d_ll_vs_elo": round(b["d_log_loss"], 5), "d_ll_ci_lo": round(b["d_ll_ci"][0], 5),
                       "d_ll_ci_hi": round(b["d_ll_ci"][1], 5), "d_brier_vs_elo": round(b["d_brier"], 5),
                       "d_br_ci_lo": round(b["d_br_ci"][0], 5), "d_br_ci_hi": round(b["d_br_ci"][1], 5),
                       "p_better_than_elo": round(b["p_model_better_ll"], 3),
                       "pregame_knowable": variant == "pregame" or n.startswith("elo")}
                if recal is not None:
                    br = Scoring.BootstrapDiff(preds[n], recal, y[s], dates[s])
                    row.update({"d_ll_vs_elo_recal": round(br["d_log_loss"], 5),
                                "d_ll_recal_ci_lo": round(br["d_ll_ci"][0], 5), "d_ll_recal_ci_hi": round(br["d_ll_ci"][1], 5)})
                metrics_rows.append(row)
                if s == HOLDOUT:
                    detail["calibration"][n] = Scoring.Calibration(preds[n], y[s])
            # margin / total accuracy of the point predictions (no market lines available)
            mm_margin = np.array([r[f"mm_margin_{variant}"] for r in by_season[s]])
            mmt_margin = np.array([r[f"mmt_margin_{variant}"] for r in by_season[s]])
            rapm_margin = np.array([r[f"rapm_margin_{variant}"] for r in by_season[s]])
            actual_margin = np.array([r["margin"] for r in by_season[s]])
            actual_total = np.array([r["total"] for r in by_season[s]])
            mm_total = np.array([r[f"sim_total_{variant}"] for r in by_season[s]]) * run["total_scale"]
            inside = np.mean([(r[f"sim_total_q10_{variant}"] <= r["total"] <= r[f"sim_total_q90_{variant}"]) for r in by_season[s]])
            inside_m = np.mean([(r[f"sim_margin_q10_{variant}"] <= r["margin"] <= r[f"sim_margin_q90_{variant}"]) for r in by_season[s]])
            detail["margin_total"][f"{variant}_{split}"] = {
                "n": len(actual_margin),
                "margin_mae_rapm_raw": float(np.mean(np.abs(rapm_margin - actual_margin))),
                "margin_mae_matchup_raw": float(np.mean(np.abs(mm_margin - actual_margin))),
                "margin_mae_matchup_team_raw": float(np.mean(np.abs(mmt_margin - actual_margin))),
                "margin_mae_zero": float(np.mean(np.abs(actual_margin))),
                "total_mae_simfamily_scaled": float(np.mean(np.abs(mm_total - actual_total))),
                "total_mae_season_mean": float(np.mean(np.abs(actual_total - actual_total.mean()))),
                "sim_total_q10_q90_coverage": float(inside), "sim_margin_q10_q90_coverage": float(inside_m)}
    for s, split in SPLITS.items():
        rs = [r for r in run["rows"] if r["season"] == s and r.get("assign_l1_full") is not None]
        if rs:
            w = np.array([r["assign_poss"] for r in rs])
            detail["assignment"][split] = {m: float(np.average([r[f"assign_l1_{m}"] for r in rs], weights=w))
                                           for m in ("full", "prior", "hist")} | {"games": len(rs)}
            detail["pairs"][split] = {k: float(np.sum([r[f"pair_ll_{k}"] for r in rs])) for k in ("pair", "offender", "league")} | {"games": len(rs)}
    detail.update({k: run[k] for k in ("best", "sim_family", "grids", "best_sim", "total_scale", "workers", "sims", "cores", "head", "secs", "configs_tried")})
    detail["tune_scores"] = run["tune_scores"]
    detail["sim_tune"] = {str(run["sim_grid"][int(k)]): v for k, v in run["sim_tune"].items()}
    detail["metrics"] = metrics_rows
    with open(os.path.join(args.out_dir, f"{BASE}.csv"), "w", newline="", encoding="utf-8") as f:
        cols = list(dict.fromkeys(k for r in metrics_rows for k in r))
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(metrics_rows)
    json.dump(detail, open(os.path.join(args.out_dir, f"{BASE}.json"), "w", encoding="utf-8"), indent=1)
    for r in metrics_rows:
        if r["split"].startswith("holdout") or r["split"].startswith("tune"):
            print(r["split"], r["series"], r["n"], r["log_loss"], r["brier"], r["accuracy"], r["ece"],
                  r["d_ll_vs_elo"], (r["d_ll_ci_lo"], r["d_ll_ci_hi"]), r.get("d_ll_vs_elo_recal"))
    print(json.dumps({k: detail[k] for k in ("assignment", "pairs", "margin_total", "best", "best_sim")}, indent=1))


if __name__ == "__main__":
    Main()
