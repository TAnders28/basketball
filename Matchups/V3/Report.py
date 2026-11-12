r"""
Builds the v3 deliverable from logs/v3 rows: per-trial scores on the 2025-26 selection split and the 2026-27 forward
test, paired bootstraps vs phase-1 Elo and kalman_team, calibration, trial registry, PBO and selection-adjusted
bound, prior sensitivity, learned-variance and stacker summaries. Writes <out>/2026-11-12_matchup-models-v3.{csv,json}
and the tables part of the .md (verdict text is added by hand after reading the numbers).
"""

import csv
import json
import os
import pickle

import numpy as np

from Matchups.V3 import Trials
from Matchups.V3.Backtest import FORWARD_FROM, SENS, SPLIT, LoadRows
from Matchups.V3.Evaluate import LossVectors, PairedBootstrap, Score
from Matchups.V3.Overfit import PboCscv, SelectionAdjusted

PHASE1_CSV = r"E:\AgentDocs\Basketbot\2026-11-12_matchup-models\2026-11-12_matchup-models.csv"
STEM = "2026-11-12_matchup-models-v3"


def Split(rows, which):
    if which == "sel":
        return [r for r in rows if SPLIT[0] <= r["game_date"] <= SPLIT[1]]
    return [r for r in rows if r["game_date"] >= FORWARD_FROM]


def Align(a, b):
    ib = {r["game_id"]: r for r in b}
    a2 = [r for r in a if r["game_id"] in ib]
    return a2, [ib[r["game_id"]] for r in a2]


def Compare(rows, base_rows):
    a, b = Align(rows, base_rows)
    if len(a) < 30:
        return {}
    la, lb = LossVectors(a), LossVectors(b)
    days = [r["game_date"] for r in a]
    out = {}
    for k in ("log_loss", "brier", "crps"):
        out[k] = PairedBootstrap(la[k], lb[k], days)
    return out


def Write(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = LoadRows()
    post = pickle.load(open(os.path.join("logs", "v3", "post.pkl"), "rb"))
    trials = dict(runs)
    trials.update(post["post"])
    elo, kal = trials["elo_phase1"]["rows"], trials["kalman_team"]["rows"]
    table = []
    for t, d in sorted(trials.items()):
        rec = {"trial": t, "approach": d["name"], "params": json.dumps(d["params"]), "oracle": d["oracle"],
               "kind": d.get("kind", "sens" if d["params"] else "base")}
        for which in ("sel", "fwd"):
            rows = Split(d["rows"], which)
            s = Score(rows)
            if not s:
                continue
            for k in ("n", "log_loss", "brier", "crps", "ece", "mcb", "dsc", "unc", "coverage80", "spread_skill",
                      "pit_max_dev", "rmse_margin", "ppc_margin_sd_pred", "ppc_margin_sd_actual", "ppc_20plus_pred",
                      "ppc_20plus_actual", "ppc_home_win_pred", "ppc_home_win_actual"):
                rec[f"{which}_{k}"] = s[k]
            rec[f"{which}_pit"] = s["pit"]
            rec[f"{which}_reliability"] = s["reliability"]
            for bname, brows in (("elo", Split(elo, which)), ("kalman", Split(kal, which))):
                c = Compare(rows, brows)
                for k, v in c.items():
                    rec[f"{which}_d{k}_vs_{bname}"] = v["diff"]
                    rec[f"{which}_d{k}_vs_{bname}_lo"] = v["lo"]
                    rec[f"{which}_d{k}_vs_{bname}_hi"] = v["hi"]
                    rec[f"{which}_p_better_{k}_vs_{bname}"] = v["p_better"]
        table.append(rec)
    by = {r["trial"]: r for r in table}

    # trial registry: phase-1 holdout series + every v3 trial scored on 2025-26
    n_p1 = Trials.SeedPhase1(PHASE1_CSV)
    for r in table:
        if "sel_n" in r:
            Trials.Append(r["trial"], {"params": r["params"], "oracle": r["oracle"]}, "2025-26",
                          {k[4:]: r[k] for k in r if k.startswith("sel_") and not isinstance(r[k], list)})
    reg = Trials.Load()
    n_phase1 = len({x["trial"] for x in reg if x["phase"] == "phase1" and x["split"] == "2025-26"})
    n_v3 = len({x["trial"] for x in reg if x["phase"] == "v3" and x["split"] == "2025-26"})

    # overfitting: legit (non-oracle) trials with full coverage of the selection games
    legit = [t for t, d in trials.items() if not d["oracle"] and "sel_n" in by[t]]
    common = set.intersection(*[{r["game_id"] for r in Split(trials[t]["rows"], "sel")} for t in legit])
    sel_rows = {t: {r["game_id"]: r for r in Split(trials[t]["rows"], "sel")} for t in legit}
    ref = sorted((sel_rows["elo_phase1"][g] for g in common), key=lambda r: (r["game_date"], r["game_id"]))
    gids = [r["game_id"] for r in ref]
    days = [r["game_date"] for r in ref]
    Lll = np.array([LossVectors([sel_rows[t][g] for g in gids])["log_loss"] for t in legit])
    Lcr = np.array([LossVectors([sel_rows[t][g] for g in gids])["crps"] for t in legit])
    pbo_ll, pbo_cr = PboCscv(Lll), PboCscv(Lcr)
    ik, ie = legit.index("kalman_team"), legit.index("elo_phase1")
    adj_k = SelectionAdjusted(Lll, Lll[ik], days)
    adj_e = SelectionAdjusted(Lll, Lll[ie], days)
    best = legit[int(np.argmin(Lll.mean(1)))]
    # forward rank of the selected trial among legit trials on the forward test
    fwd = {t: by[t].get("fwd_log_loss") for t in legit if by[t].get("fwd_log_loss") is not None}
    fwd_rank = 1 + sum(1 for v in fwd.values() if v < fwd.get(best, 9))
    over = {"n_phase1_trials": n_phase1, "n_v3_trials": n_v3, "n_total_trials_on_2025_26": n_phase1 + n_v3,
            "n_trials_in_pbo": len(legit), "n_games_in_pbo": len(gids), "pbo_log_loss": pbo_ll, "pbo_crps": pbo_cr,
            "selected": best, "selected_sel_log_loss": by[best]["sel_log_loss"], "selected_fwd_log_loss": fwd.get(best),
            "selected_fwd_rank": fwd_rank, "n_fwd_trials": len(fwd),
            "adj_vs_kalman": {k: v for k, v in adj_k.items() if k != "gains"}, "adj_vs_elo": {k: v for k, v in adj_e.items() if k != "gains"},
            "best_vs_kalman": legit[adj_k["best"]], "best_vs_elo": legit[adj_e["best"]]}

    # prior sensitivity
    sens = []
    for n, (param, base, kind) in SENS.items():
        row = {"approach": n, "param": param, "base": base}
        for t, d in trials.items():
            if d["name"] == n and not d["oracle"] and d.get("kind") not in ("post", "stack") and set(d["params"]) <= {param}:
                scale = 1.0 if not d["params"] else d["params"][param] / base
                scale = scale if kind == "var" else scale ** 2 if kind == "sd" else 1 / scale
                row[f"ll_x{scale:g}"] = by[t].get("sel_log_loss")
                row[f"crps_x{scale:g}"] = by[t].get("sel_crps")
                if d["params"]:
                    a, b = Align(Split(d["rows"], "sel"), Split(trials[n]["rows"], "sel"))
                    row[f"max_dp_x{scale:g}"] = float(max(abs(x["p_home"] - y["p_home"]) for x, y in zip(a, b))) if a else None
        sens.append(row)

    # learned variance summaries (last recorded values on the forward test)
    lv = {}
    for t, d in runs.items():
        rows = d["rows"]
        if rows and ("tau_scale" in rows[-1] or "tau_pair_hat" in rows[-1]) and not d["params"] and not d["oracle"]:
            lv[t] = {"tau_scale_last": rows[-1].get("tau_scale"), "tau_pair_last": rows[-1].get("tau_pair_hat"),
                     "dropped_share_sel": float(np.mean([r.get("term_dropped", 0) for r in Split(rows, "sel")] or [0])),
                     "dropped_share_fwd": float(np.mean([r.get("term_dropped", 0) for r in Split(rows, "fwd")] or [0]))}

    flat = [{k: v for k, v in r.items() if not isinstance(v, list)} for r in table]
    keys = sorted({k for r in flat for k in r}, key=lambda k: (k not in ("trial", "approach", "kind", "oracle", "params"), k))
    with open(os.path.join(out_dir, STEM + ".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
    with open(os.path.join(out_dir, STEM + ".json"), "w", encoding="utf-8") as fh:
        json.dump({"overfit": over, "sensitivity": sens, "learned_variance": lv, "stack_info": post["stack_info"],
                   "table": table}, fh, indent=1, default=str)
    print(json.dumps(over, indent=1, default=str))
