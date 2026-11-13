"""Polite, cached, read-only JSON GETs for the in-play pulls.

Hosts are allow-listed; every host has its own minimum gap between requests.
429/5xx back off exponentially. Nothing here can send an order.
"""

import json
import os
import random
import threading
import time
from urllib.parse import urlsplit

ALLOWED = {
    "api.elections.kalshi.com": 0.25,
    "gamma-api.polymarket.com": 0.30,
    "clob.polymarket.com": 0.25,
    "site.api.espn.com": 0.30,
    "data-api.polymarket.com": 0.30,
}
HEADERS = {"Accept": "application/json", "User-Agent": "basketball-inplay-research/1.0 (read-only)"}
_last = {}
_lock = threading.Lock()


class HttpError(Exception):
    pass


def _Wait(host):
    gap = ALLOWED[host]
    with _lock:
        now = time.monotonic()
        wait = _last.get(host, 0) + gap - now
        _last[host] = max(now, _last.get(host, 0) + gap)
    if wait > 0:
        time.sleep(wait)


def GetJson(url, params=None, tries=6, timeout=40, get=None):
    host = urlsplit(url).netloc
    if urlsplit(url).scheme != "https" or host not in ALLOWED:
        raise HttpError(f"host_not_allowed:{host}")
    if get is None:
        import requests
        get = requests.get
    delay = 2.0
    for attempt in range(tries):
        _Wait(host)
        try:
            response = get(url, params=params, headers=HEADERS, timeout=timeout, allow_redirects=False)
        except Exception as exc:  # network blip
            if attempt == tries - 1:
                raise HttpError(f"network:{type(exc).__name__}")
            time.sleep(delay + random.random())
            delay *= 2
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code in (404, 400, 422):
            raise HttpError(f"status:{response.status_code}:{response.text[:200]}")
        time.sleep(delay + random.random())
        delay = min(delay * 2, 120)
    raise HttpError(f"gave_up:{url}")


def WriteJsonAtomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, separators=(",", ":"))
    os.replace(tmp, path)


def ReadJson(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)