"""Daily NBA prop + moneyline check against the frozen 2026-11-14 prop-pricing rule.

Usage: python -m Props.DailyPropCheck [--date YYYY-MM-DD]
Prints one line: PASS (with reason and rule hash) unless a rule with status "qualified" exists.
Fail-safe: anything unexpected prints PASS. Never places orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

RULE_PATH = Path(r"E:\AgentDocs\Basketbot\2026-11-14_prop-pricing\2026-11-14_prop-pricing_rule.json")


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    a = ap.parse_args()
    try:
        rule = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    except Exception as ex:
        print(f"{a.date} NBA props/moneylines: PASS (rule file unreadable: {ex}; fail-safe)")
        return 0
    body = {k: v for k, v in rule.items() if k not in ("created_at_et", "rule_hash")}
    got = hashlib.sha1(json.dumps({**body}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if got != rule.get("rule_hash"):
        print(f"{a.date} NBA props/moneylines: PASS (rule hash mismatch {got} vs {rule.get('rule_hash')}; fail-safe)")
        return 0
    if rule.get("status") != "qualified" or rule.get("rule") is None:
        print(f"{a.date} NBA props/moneylines: PASS. No tested model clears 3c fee-adjusted edge "
              f"(prop rule {rule.get('status')}, moneyline {rule.get('moneyline_status')}; rule hash {rule['rule_hash'][:12]}; "
              f"fee = Tim's $0.02 estimate).")
        return 0
    print(f"{a.date} NBA props: a qualified rule exists ({rule['rule']}), but live slate scoring is not built yet: PASS (fail-safe).")
    return 0


if __name__ == "__main__":
    raise SystemExit(Main())
