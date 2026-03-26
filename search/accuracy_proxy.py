"""
src/search/accuracy_proxy.py
============================
Fast accuracy surrogate for the CMA-ES and greedy search engines.

Role in the pipeline
--------------------
Running full AdaRound + inference for every candidate policy during search
is prohibitively expensive (~10-20 minutes per policy).  This module trains
a small MLP that predicts the top-1 accuracy a policy would achieve in
~milliseconds, enabling hundreds of policy evaluations per search run.

Training data generation
------------------------
``ProxyDataCollector`` generates training (policy, accuracy) pairs by:

1. Sampling random policies over the pruning + bit-width space.
2. Applying each policy using fast RTN rounding (no AdaRound).
3. Evaluating accuracy on a small validation subset.
4. Recording the (feature_vector, accuracy) pair.

Approximately 100-200 random policies are sufficient for a reliable proxy
on ResNet-20/32/56.  Training takes ~2-3 hours of compute (dominated by
the fast evaluations, not the MLP fitting).

Feature vector
--------------
For each layer l the input feature vector concatenates:

    [grad_norm_l, hessian_trace_l,    ← 2 sensitivity features per layer
     pruning_ratio_l,                 ← 1 pruning decision
     weight_bits_l_2, weight_bits_l_4, weight_bits_l_8]  ← 3 one-hot bits

Total dimensionality: n_layers × 6.

MLP architecture
----------------
    Linear(n_features, 256) → ReLU → Dropout(0.1)
    Linear(256, 256)        → ReLU → Dropout(0.1)
    Linear(256, 128)        → ReLU
    Linear(128, 1)          → Sigmoid  (output ∈ [0, 1])

Output is interpreted as predicted top-1 accuracy (0–1 scale).

Usage
-----
>>> collector = ProxyDataCollector(model, sensitivity, cost_model, device)
>>> dataset = collector.collect(val_loader, n_policies=150)
>>> proxy = AccuracyProxy(n_layers=20)
>>> proxy.train(dataset)
>>> predicted_acc = proxy.predict(feature_vector)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# Candidate bit-widths (must match quant_params.yaml)
_AVAILABLE_BITS = [2, 4, 8]
_MAX_SPARSITY = 0.6   # never prune more than 60% of any layer


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_policy_features(
    layer_names: List[str],
    sensitivity: Dict[str, dict],
    policy: Dict[str, dict],
) -> torch.Tensor:
    """Encode a (sensitivity, policy) pair into a flat feature vector.

    Parameters
    ----------
    layer_names:
        Ordered list of all quantisable layer names.
    sensitivity:
        Full layer sensitivity dict from the extended profiler.
        Keys: layer name → ``{"grad_norm": float, "hessian_trace": float, ...}``
    policy:
        Policy dict, layer name →
        ``{"weight_bits": int, "activation_bits": int, "sparsity": float}``

    Returns
    -------
    torch.Tensor
        1-D float tensor of shape ``(n_layers * 6,)``.
    """
    features: List[float] = []

    for name in layer_names:
        sens = sensitivity.get(name, {})
        pol = policy.get(name, {})

        grad_norm = float(sens.get("grad_norm", 0.0))
        hessian_trace = float(sens.get("hessian_trace", 0.0))
        sparsity = float(pol.get("sparsity", 0.0))
        wb = int(pol.get("weight_bits", 8))

        # One-hot encode bit-width over {2, 4, 8}
        bits_onehot = [1.0 if wb == b else 0.0 for b in _AVAILABLE_BITS]

        features.extend([grad_norm, hessian_trace, sparsity] + bits_onehot)

    return torch.tensor(features, dtype=torch.float32)


def sample_random_policy(
    layer_names: List[str],
    available_bits: List[int] = _AVAILABLE_BITS,
    max_sparsity: float = _MAX_SPARSITY,
    first_last_bits: int = 8,
) -> Dict[str, dict]:
    """Sample a uniformly random compression policy.

    First and last layers are always kept at 8-bit, 0% sparsity (standard
    practice from HAWQ / APQ literature).
    """
    policy: Dict[str, dict] = {}
    for i, name in enumerate(layer_names):
        if i == 0 or i == len(layer_names) - 1:
            policy[name] = {"weight_bits": first_last_bits, "activation_bits": first_last_bits, "sparsity": 0.0}
        else:
            wb = random.choice(available_bits)
            sparsity = round(random.uniform(0.0, max_sparsity), 2)
            policy[name] = {"weight_bits": wb, "activation_bits": wb, "sparsity": sparsity}
    return policy


# ---------------------------------------------------------------------------
# Data collector: generate (policy, accuracy) training pairs
# ---------------------------------------------------------------------------

class ProxyDataCollector:
    """Generate training data for the accuracy proxy.

    Parameters
    ----------
    model:
        The FP32 model.  A *copy* is used for each evaluation so the
        original is never modified.
    layer_names:
        Ordered list of quantisable layer names (from cost model).
    sensitivity:
        Full sensitivity dict from the extended profiler.
    device:
        Torch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        sensitivity: Dict[str, dict],
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.layer_names = layer_names
        self.sensitivity = sensitivity
        self.device = device

    def collect(
        self,
        val_loader: DataLoader,
        n_policies: int = 150,
        eval_batches: int = 5,
        cache_path: Optional[Path] = None,
    ) -> TensorDataset:
        """Sample ``n_policies`` random policies and evaluate each quickly.

        Parameters
        ----------
        val_loader:
            Validation dataloader used for fast accuracy evaluation.
        n_policies:
            Number of random policies to sample.
        eval_batches:
            Number of validation batches used per policy evaluation.
            Fewer batches = faster but noisier accuracy estimate.
        cache_path:
            If provided, save collected (features, accuracies) tensors to
            disk so they can be reused without regenerating.

        Returns
        -------
        TensorDataset
            Dataset of (feature_vector, accuracy) pairs for proxy training.
        """
        if cache_path is not None and Path(cache_path).exists():
            logger.info("Loading proxy training data from cache: %s", cache_path)
            data = torch.load(cache_path)
            return TensorDataset(data["features"], data["accuracies"])

        logger.info("Generating %d random policies for proxy training ...", n_policies)

        all_features: List[torch.Tensor] = []
        all_accuracies: List[float] = []

        import copy

        for idx in range(n_policies):
            policy = sample_random_policy(self.layer_names)

            # Fast evaluation: apply RTN quantization (no AdaRound), measure acc
            model_copy = copy.deepcopy(self.model).to(self.device)
            self._apply_rtn_policy(model_copy, policy)
            acc = self._fast_eval(model_copy, val_loader, eval_batches)

            features = encode_policy_features(self.layer_names, self.sensitivity, policy)
            all_features.append(features)
            all_accuracies.append(acc)

            if (idx + 1) % 20 == 0:
                logger.info("  Collected %d / %d policies", idx + 1, n_policies)

        features_tensor = torch.stack(all_features)
        acc_tensor = torch.tensor(all_accuracies, dtype=torch.float32).unsqueeze(1)

        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": features_tensor, "accuracies": acc_tensor}, cache_path)
            logger.info("Proxy training data cached to %s", cache_path)

        return TensorDataset(features_tensor, acc_tensor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _apply_rtn_policy(
        self, model: nn.Module, policy: Dict[str, dict]
    ) -> None:
        """Apply Round-To-Nearest quantization in-place (fast, no AdaRound)."""
        for name, module in model.named_modules():
            if name not in policy:
                continue
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            entry = policy[name]
            wb = entry.get("weight_bits", 8)
            if wb >= 32:
                continue
            w = module.weight.data
            # Symmetric per-tensor quantization
            w_max = w.abs().max().clamp(min=1e-8)
            q_max = 2 ** (wb - 1) - 1
            scale = w_max / q_max
            w_q = (w / scale).round().clamp(-q_max - 1, q_max)
            module.weight.data = w_q * scale

    @torch.no_grad()
    def _fast_eval(
        self, model: nn.Module, loader: DataLoader, n_batches: int
    ) -> float:
        """Top-1 accuracy on the first ``n_batches`` batches of ``loader``."""
        model.eval()
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(loader):
            if i >= n_batches:
                break
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        return correct / max(total, 1)


# ---------------------------------------------------------------------------
# MLP accuracy proxy
# ---------------------------------------------------------------------------

class AccuracyProxy(nn.Module):
    """Small MLP that predicts top-1 accuracy from a policy feature vector.

    Parameters
    ----------
    n_features:
        Dimensionality of the input feature vector (``n_layers * 6``).
    hidden_dim:
        Hidden layer width.
    dropout:
        Dropout probability applied after the first two hidden layers.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, feature_vector: torch.Tensor) -> float:
        """Predict accuracy for a single feature vector.

        Parameters
        ----------
        feature_vector:
            1-D tensor of shape ``(n_features,)``.

        Returns
        -------
        float
            Predicted top-1 accuracy ∈ [0, 1].
        """
        self.eval()
        with torch.no_grad():
            x = feature_vector.unsqueeze(0)
            return self.net(x).item()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: TensorDataset,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_split: float = 0.15,
    ) -> Dict[str, List[float]]:
        """Train the proxy MLP on collected (feature, accuracy) pairs.

        Parameters
        ----------
        dataset:
            Output of ``ProxyDataCollector.collect()``.
        epochs:
            Training epochs.
        lr:
            Adam learning rate.
        batch_size:
            Mini-batch size.
        val_split:
            Fraction of data held out for validation.

        Returns
        -------
        Dict with ``"train_loss"`` and ``"val_loss"`` lists.
        """
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Train
            self.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self(x_batch)
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            # Validate
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    pred = self(x_batch)
                    val_loss += F.mse_loss(pred, y_batch).item()
            val_loss /= max(len(val_loader), 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    "Proxy training epoch %d/%d — train_loss=%.5f  val_loss=%.5f",
                    epoch + 1, epochs, train_loss, val_loss,
                )

        logger.info("Accuracy proxy training complete.")
        return history

    def save(self, path: str | Path) -> None:
        """Save proxy weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("Accuracy proxy saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load proxy weights from disk."""
        self.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info("Accuracy proxy loaded from %s", path)
