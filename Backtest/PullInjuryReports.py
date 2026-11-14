"""Resumable, polite download of official NBA injury-report PDFs.

Fetches from official.nba.com (free public). The ak-static.cms.nba.com host
listed on the official page timed out from this machine with 0 bytes.

Filename form: hourly _HHAM/_HHPM through 2025-12-21; 15-minute _HH_MMAM
from 2025-12-22. Skips existing PDFs. Rate-limited. BelowNormal.

    .venv\\Scripts\\python.exe Backtest/PullInjuryReports.py
"""
from __future__ import annotations

import json
import os
import random
import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

MAIN = Path(r"E:\Basketball\basketball")
REPORT_DIR = MAIN / "data" / "injury" / "reports"
MANIFEST = REPORT_DIR / "_manifest.jsonl"
LOG = Path(r"E:\AgentDocs\Basketbot\2026-11-13_injury-latency\_download.log")
BASE = "https://official.nba.com/referee/injury"
UA = "BasketbotInjuryResearch/1.0 (read-only; academic sports research; polite, rate-limited)"
START = date(2025, 10, 20)
END = date(2026, 11, 13)
SWITCH = date(2025, 12, 22)  # first day of 15-minute filenames
GAP_SEC = 0.25
WORKERS = 2
TIMEOUT = 45
TRIES = 4


def LimitCpu() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000
        )
    except Exception:
        pass


def Log(msg: str) -> None:
    line = f"{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def HourlyNames():
    names = []
    for ampm in ("AM", "PM"):
        for hh in range(1, 13):
            names.append(f"{hh:02d}{ampm}")
    return names


def QuarterHourNames():
    names = []
    for ampm in ("AM", "PM"):
        for hh in range(1, 13):
            for mm in (0, 15, 30, 45):
                names.append(f"{hh:02d}_{mm:02d}{ampm}")
    return names


def Planned():
    rows = []
    d = START
    while d <= END:
        form = "quarter" if d >= SWITCH else "hourly"
        slots = QuarterHourNames() if form == "quarter" else HourlyNames()
        for slot in slots:
            name = f"Injury-Report_{d.isoformat()}_{slot}.pdf"
            rows.append({"date": d.isoformat(), "slot": slot, "form": form, "name": name})
        d += timedelta(days=1)
    return rows


def LoadDone():
    """Paths already on disk or recorded as a definitive 403/404."""
    have = set()
    denied = set()
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                name = rec.get("name")
                if rec.get("ok"):
                    have.add(name)
                elif rec.get("status") in (403, 404):
                    denied.add(name)
    for path in REPORT_DIR.glob("Injury-Report_*.pdf"):
        if path.stat().st_size > 1000:
            have.add(path.name)
    # already-downloaded probes living one level up or in probe/
    for folder in (MAIN / "data" / "injury", MAIN / "data" / "injury" / "probe"):
        if not folder.exists():
            continue
        for path in folder.glob("Injury-Report_*.pdf"):
            dest = REPORT_DIR / path.name
            if path.stat().st_size > 1000:
                have.add(path.name)
                if not dest.exists():
                    REPORT_DIR.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(path.read_bytes())
    return have, denied


def AppendManifest(rec: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rec = dict(rec)
    rec.setdefault("ts", datetime.now().astimezone().isoformat(timespec="seconds"))
    with MANIFEST.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


_RATE = threading.Lock()
_LAST = 0.0

def Fetch(session: requests.Session, url: str):
    global _LAST
    delay = 2.0
    last_err = None
    for attempt in range(TRIES):
        try:
            with _RATE:
                now = time.monotonic()
                wait = _LAST + GAP_SEC - now
                _LAST = max(now, _LAST + GAP_SEC)
            if wait > 0:
                time.sleep(wait + random.random() * 0.05)
            r = session.get(url, timeout=TIMEOUT, allow_redirects=True)
            return r
        except Exception as exc:
            last_err = exc
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)
    return last_err


def Main() -> int:
    LimitCpu()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    planned = Planned()
    have, denied = LoadDone()
    todo = [r for r in planned if r["name"] not in have and r["name"] not in denied]
    Log(f"planned={len(planned)} have={len(have)} denied={len(denied)} todo={len(todo)}")
    lock = threading.Lock()
    tls = threading.local()
    counters = {"ok": 0, "err": 0, "den": 0, "done": 0}
    t0 = time.time()

    def Session():
        sess = getattr(tls, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({"User-Agent": UA, "Accept": "application/pdf,*/*"})
            tls.session = sess
        return sess

    def One(rec):
        url = f"{BASE}/{rec['name']}"
        dest = REPORT_DIR / rec["name"]
        result = Fetch(Session(), url)
        with lock:
            if isinstance(result, Exception):
                counters["err"] += 1
                AppendManifest({"name": rec["name"], "url": url, "ok": False, "error": type(result).__name__, "msg": str(result)[:200], **rec})
                Log(f"ERR {rec['name']} {type(result).__name__}: {result}")
            else:
                r = result
                is_pdf = r.status_code == 200 and r.content[:5] == b"%PDF-" and len(r.content) > 1000
                if is_pdf:
                    tmp = dest.with_suffix(".pdf.tmp")
                    tmp.write_bytes(r.content)
                    tmp.replace(dest)
                    counters["ok"] += 1
                    AppendManifest({"name": rec["name"], "url": url, "ok": True, "status": 200, "bytes": len(r.content), **rec})
                elif r.status_code in (403, 404):
                    counters["den"] += 1
                    AppendManifest({"name": rec["name"], "url": url, "ok": False, "status": r.status_code, "bytes": len(r.content), **rec})
                else:
                    counters["err"] += 1
                    AppendManifest({"name": rec["name"], "url": url, "ok": False, "status": r.status_code, "bytes": len(r.content), **rec})
                    Log(f"BAD {rec['name']} status={r.status_code} bytes={len(r.content)}")
            counters["done"] += 1
            i = counters["done"]
            if i % 50 == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed else 0
                remain = (len(todo) - i) / rate if rate else 0
                Log(f"progress {i}/{len(todo)} ok={counters['ok']} denied={counters['den']} err={counters['err']} {rate:.2f}/s remain_min={remain/60:.1f}")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(One, rec) for rec in todo]
        for fut in as_completed(futs):
            fut.result()
    Log(f"DONE ok={counters['ok']} denied={counters['den']} err={counters['err']} elapsed_min={(time.time()-t0)/60:.1f}")
    export = MAIN / "Backtest" / "ExportInjuryLatency.py"
    if export.exists():
        import subprocess
        Log("starting ExportInjuryLatency")
        rc = subprocess.call([sys.executable, str(export)], cwd=str(MAIN))
        Log(f"export exit={rc}")
        return 0 if counters["err"] == 0 and rc == 0 else 1
    return 0 if counters["err"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(Main())