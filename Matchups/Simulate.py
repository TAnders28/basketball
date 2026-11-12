"""
M3: Monte Carlo game simulator on top of the matchup model's per-pairing inputs.

Per simulated game:
  pace multiplier ~ Normal(1, PACE_SD), shared by both teams (same number of possessions)
  each pairing's points-per-possession ~ Gamma(shape, mean = model lambda)  (parameter uncertainty)
  team form ~ LogNormal(-s^2/2, s) per team (game-level hot/cold night), home edge (1 +/- h)
  team points ~ Poisson(sum_pairs poss * ppp * attribution ratio * total_scale * form * pace)
Win prob = P(home > away) + 0.5 P(tie) (overtime as a coin flip). Outputs margin and total quantiles.
"""

import numpy as np

PACE_SD = 0.035


def SimulateGame(inp: dict, configs: list, sims: int, seed: int, total_scale: float = 1.0) -> list:
    rng = np.random.default_rng(seed)
    pace = rng.normal(1.0, PACE_SD, sims)
    means = {}
    for side in ("home", "away"):
        arr = inp[side]
        poss, lam, shape = arr[:, 0], arr[:, 1], np.maximum(arr[:, 2], 0.5)
        draws = rng.gamma(shape[:, None], (lam / shape)[:, None], size=(len(lam), sims))
        means[side] = (poss[:, None] * draws).sum(0) * inp["ratio"] * total_scale * pace
    out = []
    for sigma, h in configs:
        fh = rng.lognormal(-sigma ** 2 / 2, sigma, sims)
        fa = rng.lognormal(-sigma ** 2 / 2, sigma, sims)
        hp = rng.poisson(means["home"] * fh * (1 + h))
        ap = rng.poisson(means["away"] * fa * (1 - h))
        margin, total = hp - ap, hp + ap
        out.append({"sigma": sigma, "h": h, "p_home": float((margin > 0).mean() + 0.5 * (margin == 0).mean()),
                    "margin_mean": float(margin.mean()), "margin_sd": float(margin.std()),
                    "margin_q10": float(np.percentile(margin, 10)), "margin_q90": float(np.percentile(margin, 90)),
                    "total_mean": float(total.mean()), "total_q10": float(np.percentile(total, 10)),
                    "total_q90": float(np.percentile(total, 90))})
    return out


def SimulateChunk(args):
    items, configs, sims, total_scale = args
    results = {}
    for key, inp in items:
        seed = int(key[0][-6:]) * 7 + (1 if key[1] == "oracle" else 0)
        results[key] = SimulateGame(inp, configs, sims, seed, total_scale)
    return results
