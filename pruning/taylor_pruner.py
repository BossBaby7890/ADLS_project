"""
src/pruning/taylor_pruner.py
============================
Stage 2 — **Structured Channel Pruning via Taylor Expansion**.

Reference
---------
Molchanov et al., "Pruning Convolutional Neural Networks for Resource
Efficient Inference", ICLR 2017.  https://arxiv.org/abs/1611.06440

Method
------
For each output channel c of a Conv2d layer l, the Taylor importance is:

    I(c) = |E_x[ a_c(x) · (∂L/∂a_c)(x) ]|

This is the 1st-order Taylor approximation of the increase in loss that
would result from zeroing out channel c.  Channels with low importance can
be removed with minimal accuracy impact.

The ``taylor_scores`` produced by ``src/profiler/sensitivity.py`` already
contain these values per channel — no extra backward passes are needed here.

Pruning procedure
-----------------
Given a pruning policy ``{layer_name: sparsity_ratio}``:

1. For each layer, rank channels by their Taylor score (ascending).
2. Remove the bottom ``floor(C_out * sparsity)`` channels.
3. Zero out the corresponding filters in the weight tensor and
   the matching input channels in the *next* layer (coupled pruning).
4. Apply iteratively in rounds (``n_rounds`` steps), recovering accuracy
   between rounds via KD (handled by ``kd_recovery.py``).

Coupled pruning
---------------
Removing output channels from layer l requires removing the same input
channels from layer l+1 (since the activation map dimensions must match).
This module builds a ``next_layer_map`` from the model graph so that coupled
pairs are updated atomically.

Note on batch normalisation
---------------------------
BN layers are coupled to their preceding Conv2d.  When a channel is pruned
from a Conv2d, the corresponding BN scale (gamma) and shift (beta) rows are
also zeroed — structurally equivalent to removing them.

Usage
-----
>>> from src.pruning.taylor_pruner import TaylorPruner
>>> pruner = TaylorPruner(model, taylor_scores)
>>> pruner.prune_round(policy, round_fraction=0.5)
>>> pruned_model = pruner.model
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TaylorPruner:
    """Iterative structured channel pruner using Taylor importance scores.

    Parameters
    ----------
    model:
        The FP32 model to prune in-place.
    taylor_scores:
        Dict mapping layer name (Conv2d module name) to a list of
        per-channel importance scores, as produced by the extended profiler.
        Expected format: ``{"layer1.0.conv1": [0.3, 0.8, 0.1, ...], ...}``
    skip_patterns:
        Layer name substrings that should never be pruned.
    """

    def __init__(
        self,
        model: nn.Module,
        taylor_scores: Dict[str, List[float]],
        skip_patterns: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.taylor_scores = taylor_scores
        self.skip_patterns = skip_patterns or ["bn", "shortcut", "downsample"]

        # Build (conv_name -> next_conv_name) coupling map once
        self._next_layer_map: Dict[str, str] = self._build_next_layer_map()
        # Build (conv_name -> bn_name) map for BN coupling
        self._bn_map: Dict[str, str] = self._build_bn_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune_round(
        self,
        policy: Dict[str, float],
        round_fraction: float = 1.0,
    ) -> None:
        """Apply one round of structured pruning according to ``policy``.

        Parameters
        ----------
        policy:
            Dict mapping layer name to target sparsity ratio (0.0–1.0).
            E.g. ``{"layer1.0.conv1": 0.3}`` removes 30% of channels.
        round_fraction:
            Fraction of the target sparsity to apply in this round.
            Set to ``1/n_rounds`` for iterative pruning.
            E.g. ``round_fraction=0.5`` with a target of 0.4 removes 20%
            of channels in this call.
        """
        for layer_name, target_sparsity in policy.items():
            if self._is_skipped(layer_name):
                logger.debug("Skipping pruning for %s (skip pattern match)", layer_name)
                continue

            module = self._get_module(layer_name)
            if module is None or not isinstance(module, nn.Conv2d):
                logger.warning("Layer %s not found or not Conv2d, skipping.", layer_name)
                continue

            scores = self.taylor_scores.get(layer_name)
            if scores is None:
                logger.warning(
                    "No Taylor scores for %s — skipping pruning.", layer_name
                )
                continue

            effective_sparsity = target_sparsity * round_fraction
            n_channels = module.out_channels
            n_prune = int(n_channels * effective_sparsity)

            if n_prune == 0:
                continue

            # Sort channels by importance (ascending = least important first)
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
            channels_to_prune = sorted_indices[:n_prune]
            channels_to_keep = sorted_indices[n_prune:]

            logger.info(
                "Pruning %s: removing %d / %d channels (sparsity=%.2f)",
                layer_name, n_prune, n_channels, effective_sparsity,
            )

            self._prune_output_channels(layer_name, module, channels_to_keep)
            self._prune_coupled_bn(layer_name, channels_to_keep)
            self._prune_next_layer_input(layer_name, channels_to_keep)

            # Update stored Taylor scores to reflect new channel indices
            self.taylor_scores[layer_name] = [scores[i] for i in channels_to_keep]

    def iterative_prune(
        self,
        policy: Dict[str, float],
        n_rounds: int = 4,
        recovery_fn=None,
    ) -> None:
        """Apply pruning in ``n_rounds`` steps with optional recovery between.

        Parameters
        ----------
        policy:
            Full target sparsity policy (as in ``prune_round``).
        n_rounds:
            Number of iterative pruning steps.
        recovery_fn:
            Optional callable with signature ``recovery_fn(model) -> None``
            called between rounds.  Typically wraps ``KDRecovery.recover()``.
        """
        for round_idx in range(n_rounds):
            logger.info("Pruning round %d / %d", round_idx + 1, n_rounds)
            self.prune_round(policy, round_fraction=1.0 / n_rounds)

            if recovery_fn is not None and round_idx < n_rounds - 1:
                logger.info("Running recovery after round %d", round_idx + 1)
                recovery_fn(self.model)

    # ------------------------------------------------------------------
    # Internal: weight surgery
    # ------------------------------------------------------------------

    def _prune_output_channels(
        self,
        layer_name: str,
        module: nn.Conv2d,
        channels_to_keep: List[int],
    ) -> None:
        """Remove output channels from a Conv2d weight tensor in-place."""
        keep = torch.tensor(channels_to_keep, dtype=torch.long, device=module.weight.device)

        with torch.no_grad():
            new_weight = module.weight.data.index_select(0, keep)
            module.weight = nn.Parameter(new_weight)
            if module.bias is not None:
                module.bias = nn.Parameter(module.bias.data.index_select(0, keep))

        # Update out_channels attribute
        module.out_channels = len(channels_to_keep)

    def _prune_coupled_bn(
        self,
        conv_name: str,
        channels_to_keep: List[int],
    ) -> None:
        """Zero / remove BN parameters coupled to the pruned conv."""
        bn_name = self._bn_map.get(conv_name)
        if bn_name is None:
            return

        bn_module = self._get_module(bn_name)
        if bn_module is None or not isinstance(bn_module, nn.BatchNorm2d):
            return

        keep = torch.tensor(channels_to_keep, dtype=torch.long, device=bn_module.weight.device)

        with torch.no_grad():
            bn_module.weight = nn.Parameter(bn_module.weight.data.index_select(0, keep))
            bn_module.bias = nn.Parameter(bn_module.bias.data.index_select(0, keep))
            bn_module.running_mean = bn_module.running_mean.index_select(0, keep)
            bn_module.running_var = bn_module.running_var.index_select(0, keep)

        bn_module.num_features = len(channels_to_keep)

    def _prune_next_layer_input(
        self,
        layer_name: str,
        channels_to_keep: List[int],
    ) -> None:
        """Remove input channels from the layer that follows ``layer_name``."""
        next_name = self._next_layer_map.get(layer_name)
        if next_name is None:
            return

        next_module = self._get_module(next_name)
        if next_module is None:
            return

        keep = torch.tensor(channels_to_keep, dtype=torch.long)

        with torch.no_grad():
            if isinstance(next_module, nn.Conv2d):
                keep = keep.to(next_module.weight.device)
                new_weight = next_module.weight.data.index_select(1, keep)
                next_module.weight = nn.Parameter(new_weight)
                next_module.in_channels = len(channels_to_keep)
            elif isinstance(next_module, nn.Linear):
                # After global average pooling the feature map is (B, C, 1, 1) → (B, C)
                # so the Linear input features == pruned conv output channels
                keep = keep.to(next_module.weight.device)
                new_weight = next_module.weight.data.index_select(1, keep)
                next_module.weight = nn.Parameter(new_weight)
                next_module.in_features = len(channels_to_keep)

    # ------------------------------------------------------------------
    # Internal: graph helpers
    # ------------------------------------------------------------------

    def _build_next_layer_map(self) -> Dict[str, str]:
        """Map each Conv2d/Linear to the next Conv2d/Linear in order."""
        ordered = [
            name
            for name, m in self.model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear))
        ]
        return {ordered[i]: ordered[i + 1] for i in range(len(ordered) - 1)}

    def _build_bn_map(self) -> Dict[str, str]:
        """Map each Conv2d to the immediately following BatchNorm2d."""
        modules = list(self.model.named_modules())
        bn_map: Dict[str, str] = {}
        for i, (name, module) in enumerate(modules):
            if isinstance(module, nn.Conv2d) and i + 1 < len(modules):
                next_name, next_module = modules[i + 1]
                if isinstance(next_module, nn.BatchNorm2d):
                    bn_map[name] = next_name
        return bn_map

    def _get_module(self, name: str) -> Optional[nn.Module]:
        """Retrieve a module by its dotted name."""
        parts = name.split(".")
        mod = self.model
        for part in parts:
            if not hasattr(mod, part):
                return None
            mod = getattr(mod, part)
        return mod

    def _is_skipped(self, name: str) -> bool:
        return any(pat in name for pat in self.skip_patterns)
