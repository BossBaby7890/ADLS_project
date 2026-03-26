"""
src/search/greedy_search.py
===========================
Greedy BOPs-constrained compression policy search.

Role in the pipeline
--------------------
Runs in parallel with CMA-ES inside Stage 1.5.  Produces a strong baseline
policy very quickly (minutes) via a deterministic sensitivity-ranked search.
The best of (greedy result, CMA-ES result) is passed to the pruner.

Algorithm
---------
Inspired by the mixed-precision selection strategy in HAWQ-V2 (Dong et al.,
NeurIPS 2020), Section 4.

    1. Start from an all-4-bit, zero-sparsity baseline policy.
    2. Compute the marginal sensitivity gain of promoting each layer to
       8-bit, normalised by the BOPs cost of that promotion.
       Gain/cost = (sensitivity_delta) / (bops_increase).
    3. Greedily promote the highest-ratio layer while the BOPs budget allows.
    4. After promotion rounds, demote the lowest-sensitivity layers to 2-bit
       to recover BOPs budget for pruning headroom.
    5. Independently assign per-layer sparsity ratios proportional to
       (1 - sensitivity): insensitive layers tolerate more pruning.

This greedy approach makes layer decisions sequentially and can miss
inter-layer correlations (which CMA-ES captures), but it is fast and
deterministic — useful as a comparison baseline and a search warm-start.

Usage
-----
>>> from src.search.greedy_search import GreedySearch
>>> gs = GreedySearch(layer_names, sensitivity, cost_model, proxy)
>>> policy = gs.search(bops_budget_ratio=0.35, max_sparsity=0.5)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from src.hardware.cost_model import CostModel
from src.search.accuracy_proxy import AccuracyProxy, encode_policy_features

logger = logging.getLogger(__name__)

_AVAILABLE_BITS = [2, 4, 8]
_FIRST_LAST_BITS = 8


class GreedySearch:
    """BOPs-constrained greedy mixed-precision + pruning policy search.

    Parameters
    ----------
    layer_names:
        Ordered list of quantisable layer names (from ``cost_model.get_all_layer_names()``).
    sensitivity:
        Full sensitivity dict from the extended profiler.
    cost_model:
        Initialised ``CostModel`` instance.
    proxy:
        Trained ``AccuracyProxy`` instance.
    skip_patterns:
        Layers whose names contain these patterns are left at 8-bit / 0 sparsity.
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

        # Pre-compute baseline BOPs (all-4-bit, 0 sparsity)
        self._baseline_policy = self._make_uniform_policy(bits=4, sparsity=0.0)
        self._baseline_bops = cost_model.compute_policy_bops(self._baseline_policy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        bops_budget_ratio: float = 0.35,
        max_sparsity: float = 0.5,
        sensitivity_key: str = "grad_norm",
    ) -> Dict[str, dict]:
        """Run the greedy search and return the best policy found.

        Parameters
        ----------
        bops_budget_ratio:
            Target BOPs as a fraction of the fp32 baseline.
            E.g. 0.35 = 35% of fp32 BOPs allowed.
        max_sparsity:
            Maximum per-layer pruning ratio.
        sensitivity_key:
            Which sensitivity metric to use for ranking.
            One of ``"grad_norm"`` or ``"hessian_trace"``.

        Returns
        -------
        Dict[str, dict]
            Best policy found.
            Layer name → ``{"weight_bits": int, "activation_bits": int, "sparsity": float}``
        """
        bops_budget = self.cost_model._baseline_bops * bops_budget_ratio

        # Step 1: start from all-4-bit baseline
        policy = self._make_uniform_policy(bits=4, sparsity=0.0)

        # Step 2: greedy promotion to 8-bit (highest gain/cost first)
        policy = self._greedy_promote(policy, bops_budget, sensitivity_key)

        # Step 3: greedy demotion to 2-bit (lowest sensitivity first)
        policy = self._greedy_demote(policy, bops_budget, sensitivity_key)

        # Step 4: assign sparsity ratios proportional to insensitivity
        policy = self._assign_sparsity(policy, bops_budget, max_sparsity, sensitivity_key)

        predicted_acc = self._score_policy(policy)
        bops_ratio = self.cost_model.compute_policy_bops(policy) / self.cost_model._baseline_bops
        logger.info(
            "Greedy search complete — predicted_acc=%.4f  bops_ratio=%.3f",
            predicted_acc, bops_ratio,
        )
        return policy

    # ------------------------------------------------------------------
    # Internal: greedy steps
    # ------------------------------------------------------------------

    def _greedy_promote(
        self,
        policy: Dict[str, dict],
        bops_budget: float,
        sensitivity_key: str,
    ) -> Dict[str, dict]:
        """Promote layers from 4-bit to 8-bit, highest gain/cost first."""
        promotable = [
            n for n in self.layer_names
            if not self._is_protected(n) and policy[n]["weight_bits"] == 4
        ]

        # Compute marginal gain/cost ratio for each candidate
        candidates: List[Tuple[float, str]] = []
        for name in promotable:
            sens = self._get_sensitivity(name, sensitivity_key)
            current_bops = self.cost_model.get_layer_bops(
                name, 4, 4, policy[name]["sparsity"]
            )
            promoted_bops = self.cost_model.get_layer_bops(
                name, 8, 8, policy[name]["sparsity"]
            )
            bops_increase = max(promoted_bops - current_bops, 1.0)
            ratio = sens / bops_increase
            candidates.append((ratio, name))

        # Sort descending by gain/cost
        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, name in candidates:
            # Check if promoting this layer keeps us within budget
            test_policy = {k: dict(v) for k, v in policy.items()}
            test_policy[name]["weight_bits"] = 8
            test_policy[name]["activation_bits"] = 8
            if self.cost_model.compute_policy_bops(test_policy) <= bops_budget:
                policy = test_policy
                logger.debug("Promoted %s → 8-bit", name)

        return policy

    def _greedy_demote(
        self,
        policy: Dict[str, dict],
        bops_budget: float,
        sensitivity_key: str,
    ) -> Dict[str, dict]:
        """Demote 4-bit layers to 2-bit, lowest sensitivity first."""
        current_bops = self.cost_model.compute_policy_bops(policy)

        if current_bops <= bops_budget:
            return policy  # already within budget

        demotable = [
            n for n in self.layer_names
            if not self._is_protected(n) and policy[n]["weight_bits"] == 4
        ]
        demotable.sort(key=lambda n: self._get_sensitivity(n, sensitivity_key))

        for name in demotable:
            test_policy = {k: dict(v) for k, v in policy.items()}
            test_policy[name]["weight_bits"] = 2
            test_policy[name]["activation_bits"] = 2
            policy = test_policy
            logger.debug("Demoted %s → 2-bit", name)
            if self.cost_model.compute_policy_bops(policy) <= bops_budget:
                break

        return policy

    def _assign_sparsity(
        self,
        policy: Dict[str, dict],
        bops_budget: float,
        max_sparsity: float,
        sensitivity_key: str,
    ) -> Dict[str, dict]:
        """Assign per-layer sparsity proportional to (1 - sensitivity).

        More insensitive layers get pruned more aggressively.
        Sparsity is applied only if the resulting BOPs stay within budget.
        """
        for name in self.layer_names:
            if self._is_protected(name):
                continue
            sens = self._get_sensitivity(name, sensitivity_key)
            # Insensitivity ∈ [0, 1]; scale by max_sparsity
            target_sparsity = round((1.0 - sens) * max_sparsity, 2)

            test_policy = {k: dict(v) for k, v in policy.items()}
            test_policy[name]["sparsity"] = target_sparsity
            if self.cost_model.compute_policy_bops(test_policy) <= bops_budget:
                policy = test_policy
                logger.debug("Assigned sparsity %.2f to %s", target_sparsity, name)

        return policy

    # ------------------------------------------------------------------
    # Internal: utilities
    # ------------------------------------------------------------------

    def _score_policy(self, policy: Dict[str, dict]) -> float:
        """Query the accuracy proxy for a policy."""
        features = encode_policy_features(self.layer_names, self.sensitivity, policy)
        return self.proxy.predict(features)

    def _make_uniform_policy(self, bits: int, sparsity: float) -> Dict[str, dict]:
        """All layers at the same bit-width and sparsity."""
        policy: Dict[str, dict] = {}
        for i, name in enumerate(self.layer_names):
            if i == 0 or i == len(self.layer_names) - 1:
                policy[name] = {"weight_bits": _FIRST_LAST_BITS, "activation_bits": _FIRST_LAST_BITS, "sparsity": 0.0}
            else:
                policy[name] = {"weight_bits": bits, "activation_bits": bits, "sparsity": sparsity}
        return policy

    def _get_sensitivity(self, name: str, key: str) -> float:
        return float(self.sensitivity.get(name, {}).get(key, 0.0))

    def _is_protected(self, name: str) -> bool:
        """First/last layers and skip-pattern layers are never modified."""
        if any(pat in name for pat in self.skip_patterns):
            return True
        idx = self.layer_names.index(name) if name in self.layer_names else -1
        return idx == 0 or idx == len(self.layer_names) - 1
