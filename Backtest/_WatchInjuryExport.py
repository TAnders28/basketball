"""Wait until the injury-report download has finished, then run the export."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

LOG = Path(r"E:\AgentDocs\Basketbot\2026-11-13_injury-latency\_download.log")
EXPORT = Path(r"E:\Basketball\basketball\Backtest\ExportInjuryLatency.py")
MAIN = Path(r"E:\Basketball\basketball")
PY = Path(r"E:\Basketball\basketball\.venv\Scripts\python.exe")
MARKER = Path(r"E:\AgentDocs\Basketbot\2026-11-13_injury-latency\_export_started.flag")


def LimitCpu():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


def DownloadedDone() -> bool:
    if not LOG.exists():
        return False
    text = LOG.read_text(encoding="utf-8", errors="replace")
    return "DONE ok=" in text


def Main() -> int:
    LimitCpu()
    print("watcher start", flush=True)
    while not DownloadedDone():
        time.sleep(30)
    if MARKER.exists():
        print("export already started, exiting", flush=True)
        return 0
    MARKER.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
    print("download done; starting export", flush=True)
    rc = subprocess.call([str(PY), str(EXPORT)], cwd=str(MAIN))
    print("export exit", rc, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(Main())