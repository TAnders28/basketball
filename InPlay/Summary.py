r"""Compact holdout/forward table for the final summary, with conservative CIs from ci_audit.json (no new trials)."""

import json
import os
import sys

from InPlay.Research import RES


def Table(keep):
    a = json.load(open(os.path.join(RES, "ci_audit.json")))
    rules = {}
    for k, v in a.items():
        f, ven, xm, tag, rule, sp, b = k.split("|")
        rules.setdefault((f, ven, xm, tag, rule), {})[(sp, b)] = v
    c = lambda v: f"{v['mean']:+.3f} [{v['cons_lo']:+.3f}, {v['cons_hi']:+.3f}] (n={v['n']})" if v else "-"
    p = lambda v: f"{v['mean']:+.3f}" if v else "-"
    out = ["| family | venue | tag | rule | holdout, Kalshi fee [cons. 95% CI] | holdout, Tim est. | forward, Kalshi fee [cons. 95% CI] | forward, Tim est. |",
           "|---|---|---|---|---|---|---|---|"]
    for (f, ven, xm, tag, rule), d in sorted(rules.items()):
        if not keep(f, ven, xm, tag):
            continue
        out.append(f"| {f} | {ven} | {tag.replace('_', ' ')} | {rule} | {c(d.get(('holdout', 'kalshi_taker')))} | {p(d.get(('holdout', 'tim_est_0.02')))} | "
                   f"{c(d.get(('forward', 'kalshi_taker')))} | {p(d.get(('forward', 'tim_est_0.02')))} |")
    return "\n".join(out)


if __name__ == "__main__":
    print(Table(lambda f, ven, xm, tag: xm == "taker" and (tag == "selected_on_tune" or (tag == "pre_registered" and f in ("dip_fav", "comeback", "comeback_ctrl", "late_fav")))))