"""Model-score drift monitoring (Evidently-style PSI).

Computes the Population Stability Index (PSI) between a reference score
distribution (training/champion) and the current production score
distribution. A PSI above ``0.2`` signals meaningful drift and should trigger a
retraining DAG. Implemented with NumPy so it runs without the optional
``evidently`` dependency; swap in ``evidently`` for the full HTML report.
"""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Return the PSI between two score distributions."""

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(1, ref_counts.sum()), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(1, cur_counts.sum()), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_verdict(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.2:
        return "moderate-shift"
    return "significant-drift -> trigger retraining"


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    rng = np.random.default_rng(0)
    ref = rng.beta(2, 8, 5000)
    cur = rng.beta(2.5, 7, 5000)
    psi = population_stability_index(ref, cur)
    print(f"PSI={psi:.4f} -> {drift_verdict(psi)}")
