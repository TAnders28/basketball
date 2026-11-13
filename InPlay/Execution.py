"""Fees and fills for 1-contract trades in a binary YES/NO market.

Fee bases (per traded side; no fee on settlement):
  tim_est_0.02   Tim's working estimate, $0.02 per $1 contract per side. Not a verified exchange fee.
  kalshi_taker   Kalshi published general taker fee 0.07*P*(1-P) per contract (fee schedule PDF,
                 "Last updated and effective: July 7, 2026"; KXNBAGAME multiplier 1). Unrounded per
                 contract (the round-up to a centicent is on the whole order).
  kalshi_maker   Kalshi maker fee 0.0175*P*(1-P) per contract.
  none           Frictionless (calibration-style check only).
"""

FEE_BASES = ("tim_est_0.02", "kalshi_taker", "kalshi_maker", "none")


def Fee(price, basis):
    if basis == "tim_est_0.02":
        return 0.02
    if basis == "kalshi_taker":
        return 0.07 * price * (1.0 - price)
    if basis == "kalshi_maker":
        return 0.0175 * price * (1.0 - price)
    if basis == "none":
        return 0.0
    raise ValueError(basis)


def SideQuotes(yes_bid, yes_ask, side):
    """Executable (bid, ask) for holding `side` (+1 = home YES, -1 = home NO = buy YES-complement).

    Buying NO costs 1 - yes_bid; selling NO receives 1 - yes_ask.
    """
    if yes_bid is None or yes_ask is None:
        return None, None
    if side == 1:
        return yes_bid, yes_ask
    return 1.0 - yes_ask, 1.0 - yes_bid


def TakerBuy(yes_bid, yes_ask, side, slip):
    _, ask = SideQuotes(yes_bid, yes_ask, side)
    if ask is None or ask <= 0 or ask >= 1:
        return None
    return min(0.99, ask + slip)


def TakerSell(yes_bid, yes_ask, side, slip):
    bid, _ = SideQuotes(yes_bid, yes_ask, side)
    if bid is None or bid <= 0 or bid >= 1:
        return None
    return max(0.01, bid - slip)


def TradePnl(entry, exit_price, entry_fee_basis, exit_fee_basis, settled):
    """P&L of one contract. settled=True means exit_price is the 0/1 settlement (no fee)."""
    pnl = exit_price - entry - Fee(entry, entry_fee_basis)
    if not settled:
        pnl -= Fee(exit_price, exit_fee_basis)
    return pnl