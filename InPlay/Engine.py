"""Event-driven 1-contract in-play backtest engine.

Timing (no look-ahead):
  * A signal at bar i uses only values as of bar i (and earlier) plus pregame facts.
  * Taker fills happen at bar i+LAT quotes (default LAT=1 minute later): buy at the side ask + slip,
    sell at the side bid - slip.
  * Maker entry rests at the side bid seen at bar i; it fills only if a trade prints strictly through
    that price during bars i+1..i+WAIT (Kalshi trade low/high), else it is cancelled.
  * An open position at the last in-play bar settles at the 0/1 outcome (no settlement fee).
Prices are home-YES probabilities; side +1 holds home YES, side -1 holds home NO (away).
"""

from InPlay.Execution import Fee, SideQuotes, TakerBuy, TakerSell

REG_END = 2880.0


class Game:
    """Plain-list arrays for one game's in-play bars (fast scalar access)."""

    __slots__ = ("gid", "day", "season", "stype", "venue", "pre", "won", "t", "el", "per", "sd", "wp",
                 "bid", "ask", "mid", "trlo", "trhi", "pm", "pmage", "n", "total", "spread")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))
        self.n = len(self.t)


def SideMid(m, side):
    return m if side == 1 else 1.0 - m


def _Entry(fam, P, G, i):
    m = G.mid[i]
    if m is None:
        return 0
    el = G.el[i]
    pre = G.pre
    if el < P.get("tmin", 0) or el > P.get("tmax", REG_END):
        return 0
    if P.get("total_min") is not None and (G.total is None or G.total < P["total_min"]):
        return 0
    if P.get("total_max") is not None and (G.total is None or G.total > P["total_max"]):
        return 0
    if fam == "dip_fav":
        fav = 1 if pre >= 0.5 else -1
        fp = SideMid(pre, fav)
        if P["pre_lo"] <= fp <= P["pre_hi"] and SideMid(m, fav) <= P["entry"]:
            return fav
        return 0
    if fam == "dip_any":
        for side in (1, -1):
            sm = SideMid(m, side)
            if P["entry"] - 0.10 <= sm <= P["entry"]:
                return side
        return 0
    if fam in ("fade_run", "momentum"):
        n = P["look"]
        if i - n < 0 or G.mid[i - n] is None:
            return 0
        d = m - G.mid[i - n]
        if not (P.get("band_lo", 0.1) <= m <= P.get("band_hi", 0.9)):
            return 0
        side = 0
        if d <= -P["drop"]:
            side = 1
        elif d >= P["drop"]:
            side = -1
        return side if fam == "fade_run" else -side
    if fam in ("late_fav", "late_dog"):
        if el < REG_END - P["mins"] * 60 or el > REG_END:
            return 0
        fav = 1 if m >= 0.5 else -1
        fm = SideMid(m, fav)
        if P["lo"] <= fm <= P["hi"]:
            return fav if fam == "late_fav" else -fav
        return 0
    if fam == "half_reset":
        if G.per[i] != 2 or el < 1440:
            return 0
        for side in (1, -1):
            if SideMid(m, side) <= SideMid(pre, side) - P["k"]:
                return side
        return 0
    if fam == "anchor":
        for side in (1, -1):
            if SideMid(m, side) <= SideMid(pre, side) - P["k"] and SideMid(m, side) >= P.get("floor", 0.1):
                return side
        return 0
    if fam == "espn_lag":
        w = G.wp[i]
        if w is None or not (0.1 <= m <= 0.9):
            return 0
        g = w - m
        if g >= P["gap"]:
            return 1 if P.get("dir", 1) == 1 else -1
        if g <= -P["gap"]:
            return -1 if P.get("dir", 1) == 1 else 1
        return 0
    if fam == "xvenue":
        p = G.pm[i]
        if p is None or G.pmage[i] is None or G.pmage[i] > 60 or G.ask[i] is None:
            return 0
        if p - G.ask[i] >= P["gap"]:
            return 1
        if G.bid[i] - p >= P["gap"]:
            return -1
        return 0
    if fam in ("comeback", "comeback_ctrl"):
        close = abs(pre - 0.5)
        if fam == "comeback" and close > P["close"]:
            return 0
        if fam == "comeback_ctrl" and not (P["mis_lo"] <= close <= P["mis_hi"]):
            return 0
        if P["mode"] == "price":
            for side in (1, -1):
                if P["lead_lo"] <= SideMid(m, side) <= P["lead_hi"]:
                    leader = side
                    break
            else:
                return 0
        else:
            sd = G.sd[i]
            if P["pts_lo"] <= abs(sd) <= P["pts_hi"]:
                leader = 1 if sd > 0 else -1
            else:
                return 0
        if fam == "comeback_ctrl" and P.get("leader_is") is not None:
            fav = 1 if pre >= 0.5 else -1
            if (leader == fav) != (P["leader_is"] == "fav"):
                return 0
        return -leader
    raise ValueError(fam)


def _ExitWanted(fam, P, G, i, pos):
    m = G.mid[i]
    if m is None:
        return None
    sm = SideMid(m, pos["side"])
    if "tgt" in P and P["tgt"] is not None and sm >= P["tgt"]:
        return "target"
    if "tgt_d" in P and P["tgt_d"] is not None and sm >= pos["mid0"] + P["tgt_d"]:
        return "target"
    if "stop" in P and P["stop"] is not None and sm <= P["stop"]:
        return "stop"
    if "stop_d" in P and P["stop_d"] is not None and sm <= pos["mid0"] - P["stop_d"]:
        return "stop"
    if P.get("hold") and i - pos["i"] >= P["hold"]:
        return "time"
    if P.get("t_exit") is not None and G.el[i] >= P["t_exit"]:
        return "time"
    if fam == "espn_lag" and P.get("close_gap") and G.wp[i] is not None:
        if (G.wp[i] - m) * pos["side"] * P.get("dir", 1) <= 0:
            return "converged"
    if fam == "xvenue" and P.get("close_gap") and G.pm[i] is not None:
        if (G.pm[i] - m) * pos["side"] <= 0:
            return "converged"
    if fam == "half_reset" and G.per[i] >= 4:
        return "time"
    return None


def RunGame(fam, P, G, lat=1, slip=0.005, exec_mode="taker", wait=3):
    """Return list of trade dicts (fees applied later by basis)."""
    trades = []
    pos = None
    i = 0
    max_tr = P.get("max_tr", 1)
    last = G.n - 1
    while i <= last:
        if pos is None:
            if len(trades) >= max_tr or i + lat > last:
                break
            side = _Entry(fam, P, G, i)
            if side == 0:
                i += 1
                continue
            if exec_mode == "taker":
                j = i + lat
                px = TakerBuy(G.bid[j], G.ask[j], side, slip)
                if px is None:
                    i += 1
                    continue
            else:
                bid, _ = SideQuotes(G.bid[i], G.ask[i], side)
                if bid is None or bid <= 0.01:
                    i += 1
                    continue
                px, j = None, None
                for k in range(i + 1, min(last, i + wait) + 1):
                    lo, hi = G.trlo[k], G.trhi[k]
                    if side == 1 and lo is not None and lo < bid - 1e-9:
                        px, j = bid, k
                        break
                    if side == -1 and hi is not None and hi > 1.0 - bid + 1e-9:
                        px, j = bid, k
                        break
                if px is None:
                    i += wait + 1
                    continue
            mid0 = SideMid(G.mid[i], side)
            pos = {"side": side, "i": j, "px": px, "mid0": mid0, "sig_i": i, "el": G.el[i], "maker": exec_mode != "taker"}
            i = j
            continue
        why = _ExitWanted(fam, P, G, i, pos) if i > pos["i"] else None
        if why and i + lat <= last:
            j = i + lat
            px = TakerSell(G.bid[j], G.ask[j], pos["side"], slip)
            if px is not None:
                trades.append(_Close(G, pos, px, False, why))
                pos = None
                i = j + 1
                continue
        i += 1
    if pos is not None:
        won = G.won if pos["side"] == 1 else 1 - G.won
        trades.append(_Close(G, pos, float(won), True, "settle"))
    return trades


def _Close(G, pos, exit_px, settled, why):
    return {"gid": G.gid, "day": G.day, "season": G.season, "side": pos["side"], "entry": pos["px"],
            "exit": exit_px, "settled": settled, "why": why, "el": pos["el"], "maker": pos["maker"]}


def TradePnl(tr, basis):
    entry_basis = basis
    if tr["maker"] and basis == "kalshi_taker":
        entry_basis = "kalshi_maker"
    pnl = tr["exit"] - tr["entry"] - Fee(tr["entry"], entry_basis)
    if not tr["settled"]:
        pnl -= Fee(tr["exit"], basis)
    return pnl