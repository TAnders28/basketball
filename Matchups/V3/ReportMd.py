r"""Markdown writer for the v3 deliverable (tables from the Report json; verdict/next steps from a text file)."""

import json
import os

STEM = "2026-11-12_matchup-models-v3"


def F(x, d=4):
    return "" if x is None else f"{x:.{d}f}"


def CI(r, key):
    if r.get(key) is None:
        return ""
    return f"{r[key]:+.4f} [{r[key + '_lo']:+.4f}, {r[key + '_hi']:+.4f}]"


def ResultTable(rows, which, A):
    A(f"| trial | n | log loss | dLL vs Elo [95% CI] | dLL vs kalman_team [95% CI] | Brier | CRPS | dCRPS vs kalman_team [95% CI] | ECE | cov80 | PIT max dev |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r.get(f"{which}_log_loss", 9)):
        if r.get(f"{which}_n") is None:
            continue
        A(f"| `{r['trial']}` | {r[f'{which}_n']} | {F(r[f'{which}_log_loss'])} | {CI(r, f'{which}_dlog_loss_vs_elo')} | "
          f"{CI(r, f'{which}_dlog_loss_vs_kalman')} | {F(r[f'{which}_brier'])} | {F(r[f'{which}_crps'], 3)} | "
          f"{CI(r, f'{which}_dcrps_vs_kalman')} | {F(r[f'{which}_ece'], 3)} | {F(r[f'{which}_coverage80'], 3)} | {F(r[f'{which}_pit_max_dev'], 3)} |")


def WriteMd(out_dir, intro_path, verdict_path, cpu_note):
    J = json.load(open(os.path.join(out_dir, STEM + ".json"), encoding="utf-8"))
    T = J["table"]
    by = {r["trial"]: r for r in T}
    L = []
    A = L.append
    A(open(intro_path, encoding="utf-8").read().rstrip())
    A("")
    base = [r for r in T if r["kind"] in ("base", "baseline") and not r["oracle"]]
    oracle = [r for r in T if r["oracle"]]
    stacks = [r for r in T if r["kind"] == "stack"]
    post = [r for r in T if r["kind"] == "post"]
    sens = [r for r in T if r["kind"] == "sens"]
    A("## Results: 2025-26 selection split (regular season, rows from 2025-10-01)")
    A("")
    A("Negative dLL / dCRPS = better than the baseline. 95% CIs are day-clustered paired bootstraps (2,000 reps). "
      "Phase-1 Elo has no margin model; its CRPS uses the implied Normal(13 * Phi^-1(p), 13^2), which reproduces its probabilities exactly.")
    A("")
    A("### Base approaches and baselines")
    A("")
    ResultTable(base, "sel", A)
    A("")
    A("### Stacks (held-out predictions only; weights refit weekly on strictly earlier days)")
    A("")
    ResultTable(stacks, "sel", A)
    A("")
    A("### Post-processed (EMOS = min-CRPS Normal recalibration; temp = logistic temperature), trailing 150 days, refit weekly")
    A("")
    ResultTable(post, "sel", A)
    A("")
    A("### ORACLE availability (who actually played; not knowable pregame in general). Not eligible for selection.")
    A("")
    ResultTable(oracle, "sel", A)
    A("")
    A("## Calibration (2025-26)")
    A("")
    A("CORP decomposition of the Brier score (BS = MCB - DSC + UNC): MCB = miscalibration, DSC = discrimination. PPC = posterior predictive checks of the margin distribution.")
    A("")
    A("| trial | Brier | MCB | DSC | UNC | spread/skill | margin SD pred / actual | P(|m|>=20) pred / actual | home win pred / actual | PIT deciles |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(base + stacks, key=lambda r: r.get("sel_log_loss", 9)):
        if r.get("sel_n") is None:
            continue
        A(f"| `{r['trial']}` | {F(r['sel_brier'])} | {F(r['sel_mcb'])} | {F(r['sel_dsc'])} | {F(r['sel_unc'])} | {F(r['sel_spread_skill'], 3)} | "
          f"{F(r['sel_ppc_margin_sd_pred'], 1)} / {F(r['sel_ppc_margin_sd_actual'], 1)} | {F(r['sel_ppc_20plus_pred'], 3)} / {F(r['sel_ppc_20plus_actual'], 3)} | "
          f"{F(r['sel_ppc_home_win_pred'], 3)} / {F(r['sel_ppc_home_win_actual'], 3)} | {' '.join(f'{x:.2f}' for x in r['sel_pit'])} |")
    A("")
    for t in ("elo_phase1", "kalman_team", J["overfit"]["selected"]):
        if t in by and by[t].get("sel_reliability"):
            A(f"CORP reliability, `{t}` (bin lower edge, n, mean forecast, CORP-recalibrated, observed):")
            A("")
            A("| bin | n | forecast | recalibrated | observed |")
            A("|---|---|---|---|---|")
            for b in by[t]["sel_reliability"]:
                A(f"| {b[0]:.1f} | {b[1]} | {b[2]:.3f} | {b[3]:.3f} | {b[4]:.3f} |")
            A("")
    A("## Locked forward test: 2026-27 games from 2026-10-20 (never used for any choice)")
    A("")
    ResultTable(base + stacks + [r for r in post if r["trial"].startswith(("kalman_team", "stack"))] + oracle, "fwd", A)
    A("")
    o = J["overfit"]
    A("## Trial count and overfitting")
    A("")
    A(f"- Trials scored on the 2025-26 holdout: **{o['n_total_trials_on_2025_26']}** = {o['n_phase1_trials']} phase-1 series (from the phase-1 csv) + {o['n_v3_trials']} v3 trials (base, sensitivity, post-processed, stacks, oracle). Registry: `logs/v3/trials.jsonl`, copied here as `{STEM}_trials.jsonl`.")
    A(f"- PBO by CSCV (16 date blocks, {o['pbo_log_loss']['n_splits']:,} splits, {o['n_trials_in_pbo']} non-oracle v3 trials with per-game rows, {o['n_games_in_pbo']} games): **PBO(log loss) = {o['pbo_log_loss']['pbo']:.3f}**, PBO(CRPS) = {o['pbo_crps']['pbo']:.3f}. Phase-1 series have no per-game rows here and are only counted, not in the PBO matrix.")
    A(f"- In-sample winner on 2025-26 (lowest log loss): `{o['selected']}` ({F(o['selected_sel_log_loss'])}); on the forward test it scores {F(o['selected_fwd_log_loss'])} and ranks {o['selected_fwd_rank']} of {o['n_fwd_trials']}.")
    for lab, key, bt in (("kalman_team", "adj_vs_kalman", "best_vs_kalman"), ("phase-1 Elo", "adj_vs_elo", "best_vs_elo")):
        a = o[key]
        A(f"- Selection-adjusted (max-statistic bootstrap over all {o['n_trials_in_pbo']} trials) log-loss gain of the best trial `{o[bt]}` vs {lab}: gain {a['gain']:+.4f}, naive 95% CI [{a['naive_lo']:+.4f}, {a['naive_hi']:+.4f}], **simultaneous lower bound {a['adjusted_lo']:+.4f}**.")
    A("")
    A("## Prior sensitivity (2025-26 log loss / CRPS at 0.5x, 1x, 2x the main prior variance)")
    A("")
    A("| approach | knob (base) | LL x0.5 | LL x1 | LL x2 | CRPS x0.5 | CRPS x1 | CRPS x2 | max abs dp x0.5 | max abs dp x2 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for s in J["sensitivity"]:
        A(f"| `{s['approach']}` | {s['param']} ({s['base']:g}) | {F(s.get('ll_x0.5'))} | {F(s.get('ll_x1'))} | {F(s.get('ll_x2'))} | "
          f"{F(s.get('crps_x0.5'), 3)} | {F(s.get('crps_x1'), 3)} | {F(s.get('crps_x2'), 3)} | {F(s.get('max_dp_x0.5'), 3)} | {F(s.get('max_dp_x2'), 3)} |")
    A("")
    A("## Learned interaction / coach-term variance (evidence fixed point; family dropped when the scale < 0.05 or pair sd < 0.1 per 100)")
    A("")
    A("| trial | last prior-variance scale | last pair sd (per 100) | share of 2025-26 games dropped | share of forward games dropped |")
    A("|---|---|---|---|---|")
    for t, v in sorted(J["learned_variance"].items()):
        A(f"| `{t}` | {F(v['tau_scale_last'], 3)} | {F(v['tau_pair_last'], 3)} | {F(v['dropped_share_sel'], 2)} | {F(v['dropped_share_fwd'], 2)} |")
    A("")
    A("## Stacker weights (last fit)")
    A("")
    for lab, info in J["stack_info"].items():
        names = info["members"]
        top = sorted(zip(names, info["w_global"]), key=lambda x: -x[1])[:6]
        A(f"- `{lab}` global: " + ", ".join(f"{n} {w:.2f}" for n, w in top))
        for c, w in sorted(info["w_context"].items()):
            topc = sorted(zip(names, w), key=lambda x: -x[1])[:4]
            A(f"  - context `{c}`: " + ", ".join(f"{n} {x:.2f}" for n, x in topc))
    A("")
    A("## CPU")
    A("")
    A(cpu_note)
    A("")
    A(open(verdict_path, encoding="utf-8").read().rstrip())
    A("")
    open(os.path.join(out_dir, STEM + ".md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
