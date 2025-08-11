import math
from typing import List, Dict, Tuple


class SimpleLogisticRegression:
    def __init__(self, learning_rate: float = 0.1, num_iterations: int = 2000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights: Tuple[float, float] | None = None  # (w0 intercept, w1 for discount)

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def fit(self, discounts: List[float], win_rates: List[float]) -> None:
        # Convert win_rates to binary target using threshold 0.5 to mimic previous behavior
        targets = [1.0 if wr >= 0.5 else 0.0 for wr in win_rates]
        w0, w1 = 0.0, 0.0

        for _ in range(self.num_iterations):
            grad_w0 = 0.0
            grad_w1 = 0.0
            for x, y in zip(discounts, targets):
                z = w0 + w1 * x
                p = self._sigmoid(z)
                error = y - p
                grad_w0 += error
                grad_w1 += error * x
            n = float(len(discounts))
            w0 += self.learning_rate * (grad_w0 / n)
            w1 += self.learning_rate * (grad_w1 / n)

        self.weights = (w0, w1)

    def predict_proba(self, discounts: List[float]) -> List[Tuple[float, float]]:
        assert self.weights is not None, "Model is not fitted"
        w0, w1 = self.weights
        probabilities = []
        for x in discounts:
            p1 = self._sigmoid(w0 + w1 * x)
            probabilities.append((1.0 - p1, p1))
        return probabilities


a = SimpleLogisticRegression


def train_model(records: List[Dict[str, float]]) -> SimpleLogisticRegression:
    discounts = [row["discount"] for row in records]
    win_rates = [row["win_rate"] for row in records]
    model = SimpleLogisticRegression(learning_rate=0.5, num_iterations=1000)
    model.fit(discounts, win_rates)
    return model


def recommend_price(model: SimpleLogisticRegression, deal_size: float, cost_basis: float) -> Dict[str, float]:
    best_score = -float("inf")
    best_scenario: Dict[str, float] = {}

    candidate_discounts = [i / 100.0 for i in range(0, 41)]  # 0.00 to 0.40 step 0.01
    for discount in candidate_discounts:
        predicted_win_rate = model.predict_proba([discount])[0][1]
        revenue = deal_size * (1.0 - discount)
        margin = revenue - cost_basis
        score = margin * predicted_win_rate

        if score > best_score:
            best_score = score
            best_scenario = {
                "discount": round(discount * 100.0, 2),
                "predicted_win_rate": round(predicted_win_rate, 2),
                "expected_margin": round(margin, 2),
                "score": round(score, 2),
            }

    return best_scenario
