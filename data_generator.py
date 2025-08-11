import random
from typing import List, Dict


def generate_deal_data(num_rows: int = 1000) -> List[Dict[str, float]]:
    """Generate synthetic B2B deal data without external dependencies.

    Returns a list of dicts with keys: 'deal_size', 'discount', 'cost_basis', 'win_rate'.
    """
    random.seed(42)

    records: List[Dict[str, float]] = []
    for _ in range(num_rows):
        deal_size = random.uniform(5000.0, 100000.0)
        discount = random.uniform(0.0, 0.4)
        cost_basis_multiplier = random.uniform(0.5, 0.8)
        cost_basis = deal_size * cost_basis_multiplier
        # Base win rate model with noise
        win_rate = 0.3 + 0.5 * discount + random.gauss(0.0, 0.05)
        # Clamp to [0, 1]
        win_rate = max(0.0, min(1.0, win_rate))

        records.append({
            "deal_size": deal_size,
            "discount": discount,
            "cost_basis": cost_basis,
            "win_rate": win_rate,
        })

    return records
