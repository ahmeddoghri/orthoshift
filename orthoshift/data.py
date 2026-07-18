from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    x1: float
    x2: float
    treatment: float
    outcome: float
    true_tau: float


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_dataset(n: int = 900, seed: int = 13, shift: float = 0.0) -> list[Sample]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x1 = rng.gauss(shift, 1.0)
        x2 = rng.gauss(-0.4 * shift, 1.0)
        prop = sigmoid(-0.25 + 1.15 * x1 - 0.85 * x2 + 0.45 * shift)
        treatment = 1.0 if rng.random() < prop else 0.0
        tau = 1.6 + 0.55 * (1.0 if x1 > 0 else -0.2)
        base = 0.8 + 1.4 * x1 - 1.1 * x2 + 0.35 * x1 * x2 + 0.25 * x1 * x1
        outcome = base + tau * treatment + rng.gauss(0.0, 0.6)
        rows.append(Sample(x1, x2, treatment, outcome, tau))
    return rows
