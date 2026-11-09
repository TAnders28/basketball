"""American-odds helpers: presented odds -> implied probability, de-vig, and bet payouts.
Same helpers as football. Free NBA game logs carry no moneylines, so the engine only uses these
when a game record has home_moneyline / away_moneyline (e.g. once an odds history is merged in).

Prediction-market edge uses Tim's working fee estimate of $0.02 per $1 contract
(fee_basis=tim_estimate_0.02_per_contract) — not a verified exchange fee.
Without prices: report accuracy only; no P&L.
"""

from typing import Optional, Tuple

# Tim's working estimate — not a verified Polymarket/Kalshi fee.
FEE_PER_CONTRACT = 0.02
FEE_BASIS = "tim_estimate_0.02_per_contract"


def ImpliedProb(american: float) -> float:
    """Raw implied probability of American odds, vig included. -198 -> 0.664, +164 -> 0.379."""
    if american < 0:
        return -american / (-american + 100.0)
    return 100.0 / (american + 100.0)


def FairProbs(home_ml: float, away_ml: float) -> Tuple[float, float]:
    """Remove the bookmaker's vig by scaling both sides so they sum to 1 (proportional de-vig)."""
    home, away = ImpliedProb(home_ml), ImpliedProb(away_ml)
    total = home + away
    return home / total, away / total


def Profit(stake: float, american: float) -> float:
    """Profit (not including the returned stake) if a bet at these odds WINS."""
    return stake * (american / 100.0 if american > 0 else 100.0 / -american)


def Settle(stake: float, american: float, won: Optional[bool]) -> float:
    """Bankroll change for a settled moneyline bet. won=None means push (stake returned)."""
    if won is None:
        return 0.0
    return Profit(stake, american) if won else -stake


def Edge(model_prob: float, ask: float, fee: float = FEE_PER_CONTRACT) -> float:
    """model_prob - ask - fee. Positive means model clears ask plus Tim's fee estimate."""
    return float(model_prob) - float(ask) - float(fee)


def BreakevenAsk(ask: float, fee: float = FEE_PER_CONTRACT) -> float:
    """Ask price a model must beat after fee (ask + fee)."""
    return float(ask) + float(fee)


def MarketFavorite(home_prob: float, away_prob: float) -> str:
    """Baseline: side with higher (fair) market probability; ties -> home."""
    return "home" if home_prob >= away_prob else "away"


def ClosingLineValue(model_side_prob: float, closing_ask: float, fee: float = FEE_PER_CONTRACT) -> dict:
    """CLV vs closing ask when a price exists. Accuracy-only callers should skip this."""
    edge = Edge(model_side_prob, closing_ask, fee=fee)
    return {
        "closing_ask": float(closing_ask),
        "model_prob": float(model_side_prob),
        "fee": float(fee),
        "fee_basis": FEE_BASIS,
        "edge_vs_close": edge,
        "beats_close_after_fee": edge > 0,
    }


def ReportWithoutPrices(correct: int, total: int) -> dict:
    """Accuracy-only summary when moneylines / asks are absent — never invents P&L."""
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else None,
        "profit": None,
        "roi": None,
        "note": "no prices; accuracy only",
    }
