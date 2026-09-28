"""Small numerical utilities."""

from __future__ import annotations

import math
from typing import Iterable


def entropy(probabilities: Iterable[float]) -> float:
    total = 0.0
    for p in probabilities:
        if p > 0.0:
            total -= p * math.log(p)
    return total


def effective_number(entropy_value: float) -> float:
    return math.exp(entropy_value)
