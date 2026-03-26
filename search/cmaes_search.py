"""
src/search/cmaes_search.py
==========================
CMA-ES joint compression policy search.

Role in the pipeline
--------------------
Runs in parallel with the greedy search inside Stage 1.5.  Searches the
joint (pruning ratio, bit-width) space across all layers simultaneously,
learning inter-layer correlations that the greedy approach misses.

CMA-ES overview
---------------
Covariance Matrix Adaptation Evolution Strategy (Hansen, 2001) maintains a
multivariate Gaussian N(m, σ²C) over the search space and iteratively
updates its mean m and covariance C by:

  1. Sampling a population of λ candidate vectors from the current Gaussian.
  2. Evaluating each candidate (via proxy + cost model — no real inference).
  3. Keeping the top-μ candidates (truncation selection).
  4. Updating m toward the weighted centroid of the top candidates.
  5. Adapting C to capture the covariance structure of good solutions.
  6. Adapting σ (step size) via the cumulative step-size adaptation (CSA).

This implementation uses the ``cma`` library (pip install cma) which
provides a production-grade CMA-ES with all standard adaptations.

Search vector
-------------
For a model with N quantisable layers, the search vector has 2N dimensions:

    x = [p_1, ..., p_N,  b_1, ..., b_N]

where:
    p_i ∈ [0, max_sparsity]  (pruning ratio for layer i)
    b_i ∈ [0, 1]             (continuous bit-width proxy, snapped to {2,4,8})

Bit-width snapping:
    b_i < 0.33   → 2-bit
    b_i < 0.67   → 4-bit
    b_i >= 0.67  → 8-bit

First and last layers are excluded from the search vector and always set to
8-bit / 0-sparsity.

Objective function
------------------
The CMA-ES *minimises* the objective, so we negate predicted accuracy and
add a large penalty for BOPs budget violations:

    f(x) = -proxy(x) + penalty(x)

    penalty(x) = max(0, bops_ratio(x) - budget_ratio) * PENALTY_WEIGHT

This is a soft constraint: policies that slightly exceed the budget get a
proportional penalty rather than being completely rejected, which helps
CMA-ES navigate the constraint boundary.

Usage
-----
>>> from src.search.cmaes_search import CMAESSearch
>>> searcher = CMAESSearch(layer_names, sensitivity, cost_model, proxy)
>>> policy = searcher.search(bops_budget_ratio=0.35, max_sparsity=0.5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.hardware.cost_model import CostModel
from src.search.accuracy_proxy import AccuracyProxy, encode_policy_features

logger = logging.getLogger(__name__)

_AVAILABLE_BITS = [2, 4, 8]
_FIRST_LAST_BITS = 8
_PENALTY_WEIGHT = 10.0   # multiplier on BOPs constraint violation


def _snap_bits(b_continuous: float) -> int:
    """Map a continuous value in [0, 1] to the nearest bit-width in {2, 4, 8}."""
    if b_continuous < 0.33:
        return 2
    elif b_continuous < 0.67:
        return 4
    else:
        return 8


class CMAESSearch:
    """CMA-ES joint search over pruning ratios and bit-widths.

    Parameters
    ----------
    layer_names:
        Ordered list of quantisable layer names.
    sensitivity:
        Full sensitivity dict from the extended profiler.
    cost_model:
        Initialised ``CostModel`` instance.
    proxy:
        Trained ``AccuracyProxy`` instance.
    skip_patterns:
        Layers left at 8-bit / 0-sparsity (not included in search vector).
    """

    def __init__(
        self,
        layer_names: List[str],
        sensitivity: Dict[str, dict],
        cost_model: CostModel,
        proxy: AccuracyProxy,
        skip_patterns: Optional[List[str]] = None,
    ) -> None:
        self.layer_names = layer_names
        self.sensitivity = sensitivity
        self.cost_model = cost_model
        self.proxy = proxy
        self.skip_patterns = skip_patterns or ["bn", "shortcut", "downsample"]

        # Identify layers that participate in the search
        self._search_layers = [
            n for i, n in enumerate(layer_names)
            if not self._is_protected(n, i)
        ]
        self._n_search = len(self._search_layers)
        # Total search vector dimension: sparsity + bit-width per search layer
        self._dim = 2 * self._n_search

        logger.info(
            "CMAESSearch initialised: %d search layers, %d-dim search space",
            self._n_search, self._dim,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        bops_budget_ratio: float = 0.35,
        max_sparsity: float = 0.5,
        popsize: int = 20,
        max_generations: int = 150,
        sigma0: float = 0.3,
        seed: int = 42,
    ) -> Dict[str, dict]:
        """Run CMA-ES and return the best policy found.

        Parameters
        ----------
        bops_budget_ratio:
            Maximum allowed BOPs as a fraction of fp32 baseline.
        max_sparsity:
            Maximum per-layer pruning ratio clipped into search vector.
        popsize:
            CMA-ES population size λ (candidates per generation).
        max_generations:
            Maximum number of generations.
        sigma0:
            Initial step size σ₀.  Controls initial exploration radius.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        Dict[str, dict]
            Best policy found by CMA-ES.
        """
        try:
            import cma
        except ImportError as e:
            raise ImportError(
                "CMA-ES requires the 'cma' package. "
                "Install it with: pip install cma"
            ) from e

        np.random.seed(seed)

        # Initial mean: all layers at 4-bit (b=0.5), 20% sparsity
        x0 = np.array(
            [0.2] * self._n_search   # pruning ratios
            + [0.5] * self._n_search  # bit-width proxies (→ 4-bit)
        )

        # Bounds: sparsity ∈ [0, max_sparsity], bits ∈ [0, 1]
        lower = [0.0] * self._n_search + [0.0] * self._n_search
        upper = [max_sparsity] * self._n_search + [1.0] * self._n_search

        cma_options = {
            "seed": seed,
            "popsize": popsize,
            "maxiter": max_generations,
            "bounds": [lower, upper],
            "verbose": -9,  # suppress cma internal output; we log ourselves
            "tolx": 1e-5,
            "tolfun": 1e-5,
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, cma_options)

        best_score = float("inf")
        best_policy: Optional[Dict[str, dict]] = None
        generation = 0

        logger.info(
            "CMA-ES search started: dim=%d  popsize=%d  max_gen=%d",
            self._dim, popsize, max_generations,
        )

        while not es.stop():
            solutions = es.ask()  # list of λ candidate vectors

            fitness_values = []
            for x in solutions:
                policy = self._decode(x, max_sparsity)
                score = self._objective(policy, bops_budget_ratio)
                fitness_values.append(score)

            es.tell(solutions, fitness_values)

            # Track best
            best_idx = int(np.argmin(fitness_values))
            if fitness_values[best_idx] < best_score:
                best_score = fitness_values[best_idx]
                best_policy = self._decode(solutions[best_idx], max_sparsity)

            generation += 1
            if generation % 10 == 0:
                bops_r = self.cost_model.bops_ratio(best_policy)
                predicted_acc = -best_score + max(
                    0.0, (bops_r - bops_budget_ratio) * _PENALTY_WEIGHT
                )
                logger.info(
                    "Gen %3d — best_score=%.5f  pred_acc≈%.4f  bops_ratio=%.3f",
                    generation, best_score, predicted_acc, bops_r,
                )

        if best_policy is None:
            logger.warning("CMA-ES did not converge — returning baseline policy.")
            best_policy = self._make_baseline_policy()

        final_acc = self.proxy.predict(
            encode_policy_features(self.layer_names, self.sensitivity, best_policy)
        )
        final_bops = self.cost_model.bops_ratio(best_policy)
        logger.info(
            "CMA-ES search complete — predicted_acc=%.4f  bops_ratio=%.3f",
            final_acc, final_bops,
        )
        return best_policy

    # ------------------------------------------------------------------
    # Internal: decode + objective
    # ------------------------------------------------------------------

    def _decode(
        self, x: np.ndarray, max_sparsity: float
    ) -> Dict[str, dict]:
        """Decode a raw CMA-ES search vector into a compression policy.

        The vector has layout:
            x[:n_search]  = sparsity ratios (clipped to [0, max_sparsity])
            x[n_search:]  = continuous bit-width proxies (snapped to {2,4,8})
        """
        policy: Dict[str, dict] = {}

        # Protected layers (first, last, skip patterns)
        for i, name in enumerate(self.layer_names):
            if self._is_protected(name, i):
                policy[name] = {
                    "weight_bits": _FIRST_LAST_BITS,
                    "activation_bits": _FIRST_LAST_BITS,
                    "sparsity": 0.0,
                }

        # Search layers
        sparsities = np.clip(x[: self._n_search], 0.0, max_sparsity)
        bit_proxies = np.clip(x[self._n_search :], 0.0, 1.0)

        for j, name in enumerate(self._search_layers):
            wb = _snap_bits(float(bit_proxies[j]))
            sp = float(round(sparsities[j], 3))
            policy[name] = {"weight_bits": wb, "activation_bits": wb, "sparsity": sp}

        return policy

    def _objective(
        self, policy: Dict[str, dict], bops_budget_ratio: float
    ) -> float:
        """CMA-ES objective: minimise negative predicted accuracy + BOPs penalty."""
        features = encode_policy_features(self.layer_names, self.sensitivity, policy)
        pred_acc = self.proxy.predict(features)

        bops_r = self.cost_model.bops_ratio(policy)
        violation = max(0.0, bops_r - bops_budget_ratio)
        penalty = violation * _PENALTY_WEIGHT

        return -pred_acc + penalty

    def _make_baseline_policy(self) -> Dict[str, dict]:
        """All-4-bit, 20% sparsity baseline."""
        policy: Dict[str, dict] = {}
        for i, name in enumerate(self.layer_names):
            if self._is_protected(name, i):
                policy[name] = {"weight_bits": 8, "activation_bits": 8, "sparsity": 0.0}
            else:
                policy[name] = {"weight_bits": 4, "activation_bits": 4, "sparsity": 0.2}
        return policy

    def _is_protected(self, name: str, idx: int) -> bool:
        if any(pat in name for pat in self.skip_patterns):
            return True
        return idx == 0 or idx == len(self.layer_names) - 1
