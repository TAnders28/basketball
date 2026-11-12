r"""
Trial registry (reading: Bailey et al. on backtest overfitting; Harvey-Liu; Gelman-Loken "garden of forking paths").

Every scored model variant is appended to logs/v3/trials.jsonl with a config hash, the split it was scored on and
headline metrics. The count of distinct trials scored on the selection split (2025-26) - including the phase-1
series scored on the same holdout - is what the overfitting estimates condition on.
"""

import hashlib
import json
import os
from datetime import datetime

PATH = os.path.join("logs", "v3", "trials.jsonl")


def Hash(name, params):
    return hashlib.sha1(json.dumps([name, params], sort_keys=True, default=str).encode()).hexdigest()[:10]


def Append(name, params, split, metrics, phase="v3", path=PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"), "trial": Hash(name, params), "name": name,
           "params": params, "phase": phase, "split": split,
           **{k: metrics.get(k) for k in ("n", "log_loss", "brier", "crps", "ece")}}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def Load(path=PATH):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def Count(split, path=PATH):
    return len({(r["phase"], r["trial"]) for r in Load(path) if r["split"] == split})


def SeedPhase1(csv_path, path=PATH):
    """Register phase-1 series scored on the 2025-26 holdout (aggregates only; no per-game rows for PBO)."""
    import csv
    have = {(r["phase"], r["trial"]) for r in Load(path)}
    n = 0
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        if not r["split"].startswith("holdout"):
            continue
        h = Hash(r["series"], {})
        if ("phase1", h) in have:
            continue
        Append(r["series"], {}, "2025-26", {"n": int(r["n"]), "log_loss": float(r["log_loss"]), "brier": float(r["brier"]),
                                             "ece": float(r["ece"])}, phase="phase1", path=path)
        have.add(("phase1", h))
        n += 1
    return n
