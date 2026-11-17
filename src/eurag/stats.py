"""Bootstrap confidence intervals over questions (no API cost)."""

import numpy as np


def bootstrap_ci(values: list[float | None], n: int = 10_000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile bootstrap interval; None values are dropped."""
    data = np.array([v for v in values if v is not None], dtype=float)
    if data.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(n, data.size), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return float(data.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_diff_ci(a: list[float | None], b: list[float | None], **kw) -> tuple[float, float, float]:
    """Interval for mean(a - b) over questions scored in both lists, resampling questions."""
    diffs = [x - y for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    return bootstrap_ci(diffs, **kw)
