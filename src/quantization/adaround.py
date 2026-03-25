"""
src/quantization/adaround.py
============================
Technique B — **AdaRound: Adaptive Rounding for Post-Training Quantization**

Reference
---------
Nagel et al., "Up or Down? Adaptive Rounding for Post-Training Quantization",
ICML 2020.  https://arxiv.org/abs/2004.10568

Overview
--------
Standard Round-to-Nearest (RTN) quantization independently clips each weight
to the nearest integer grid point.  AdaRound replaces this with a **learned**
rounding decision: for every weight w_i we introduce a continuous variable
V_i and optimise it so that the layer's *output* (not just the weights) is
preserved on a small calibration dataset.

The key insight is that rounding decisions interact — rounding one weight up
may allow a neighbouring weight to round down, reducing the total output
error.  RTN misses this because it is purely local.

Mathematical formulation
------------------------
Given a quantization grid with step size ``Δ = (x_max - x_min) / (2^b - 1)``:

  w_scaled  = w / Δ - z            (shift to integer grid, z = zero-point)

  w_floor   = floor(w_scaled)      (lower integer neighbour)

  h(V)      = clip( σ(V) * (ζ - γ) + γ, 0, 1 )   ← "rectified sigmoid"
               where σ is the logistic function,
               ζ = 1.1,  γ = −0.1   (stretch beyond [0,1] so h saturates)

  w_hat     = Δ * ( w_floor + h(V) ) + z * Δ       (quantized weight)

The objective minimised over V is:

  L(V) = ||W·x  −  W_hat(V)·x||_F²   +   λ · R_β(V)

where the regulariser R_β(V) anneals rounding decisions to {0, 1}:

  R_β(V) = Σ_i  1 − | 2·h(V_i) − 1 |^β

  β is increased linearly from β_start to β_end over the optimisation,
  making the penalty progressively sharper until every h(V_i) ∈ {0, 1}.

Integration with the APQ-Lite pipeline
---------------------------------------
After the bit-width allocator (Stage 2) assigns a bit-width to a layer, call
``AdaRoundOptimizer.optimize_layer()`` with a small calibration batch before
handing the config to ``MaseConfigGenerator`` (Stage 3).  Then use
``get_rounded_weights()`` to snap the learned rounding back into the layer.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rectified-sigmoid hyper-parameters  (Nagel et al., Appendix A)
# ---------------------------------------------------------------------------
_ZETA: float = 1.1   # upper stretch of the sigmoid
_GAMMA: float = -0.1  # lower stretch of the sigmoid


# ---------------------------------------------------------------------------
# Helper: uniform symmetric/asymmetric quantization scale & zero-point
# ---------------------------------------------------------------------------

def _compute_scale_zeropoint(
    weight: torch.Tensor,
    n_bits: int,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-tensor (scale Δ, zero-point z) for ``weight``.

    Symmetric:   z = 0,  Δ = max(|w|) / (2^(b-1) - 1)
    Asymmetric:  Δ = (max - min) / (2^b - 1),  z = round(-min / Δ)
    """
    if symmetric:
        w_max = weight.abs().max()
        q_max = 2 ** (n_bits - 1) - 1
        scale = w_max / q_max
        zero_point = torch.zeros(1, device=weight.device, dtype=weight.dtype)
    else:
        w_min = weight.min()
        w_max = weight.max()
        q_levels = 2 ** n_bits - 1
        scale = (w_max - w_min) / q_levels
        zero_point = torch.round(-w_min / scale)
        zero_point = zero_point.to(weight.dtype)
    # Guard against degenerate (all-zero) tensors
    scale = scale.clamp(min=1e-8)
    return scale, zero_point


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class AdaRoundOptimizer:
    """Layer-wise adaptive rounding optimiser.

    Usage
    -----
    >>> opt = AdaRoundOptimizer(n_bits=4)
    >>> opt.optimize_layer(layer, calibration_inputs, n_steps=500)
    >>> rounded_weights = opt.get_rounded_weights()   # drop into the graph

    Parameters
    ----------
    n_bits:
        Target quantization bit-width (e.g. 2, 4, 8).
    symmetric:
        Whether to use symmetric (z=0) or asymmetric quantization.
    reg_lambda:
        Overall weight λ on the regularisation term R_β(V).
    beta_start:
        Initial β for the annealing schedule (soft penalty, ≈ 2).
    beta_end:
        Final β for the annealing schedule (hard penalty, ≈ 20).
    learning_rate:
        Adam learning rate for V.
    """

    def __init__(
        self,
        n_bits: int = 4,
        symmetric: bool = True,
        reg_lambda: float = 1e-3,
        beta_start: float = 2.0,
        beta_end: float = 20.0,
        learning_rate: float = 1e-3,
    ) -> None:
        self.n_bits = n_bits
        self.symmetric = symmetric
        self.reg_lambda = reg_lambda
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learning_rate = learning_rate

        # Set after optimize_layer() completes
        self._V: Optional[nn.Parameter] = None
        self._scale: Optional[torch.Tensor] = None
        self._zero_point: Optional[torch.Tensor] = None
        self._weight_floor: Optional[torch.Tensor] = None  # floor(w_scaled)
        self._original_weight: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize_layer(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        n_steps: int = 500,
    ) -> None:
        """Run the AdaRound optimisation for a single layer.

        The main layer weights are **frozen**; only the rounding variables V
        are updated.  After this call, ``get_rounded_weights()`` returns the
        optimal quantized weights.

        Parameters
        ----------
        layer:
            A ``nn.Conv2d`` or ``nn.Linear`` (or any layer with ``.weight``).
        x:
            Calibration input activations, shape compatible with ``layer``.
            Must already be on the same device as ``layer``.
        n_steps:
            Number of Adam optimisation steps (paper uses 10 000; 500 is a
            fast approximation suitable for a lightweight pipeline).
        """
        if not hasattr(layer, "weight") or layer.weight is None:
            raise ValueError("Layer has no weight tensor — cannot AdaRound.")

        device = layer.weight.device
        x = x.to(device)

        # ---- 1. Freeze layer weights -----------------------------------
        # We want gradients only through V, not through the original weights.
        layer.weight.requires_grad_(False)
        if layer.bias is not None:
            layer.bias.requires_grad_(False)

        # ---- 2. Record the FP32 reference output  W·x  ----------------
        with torch.no_grad():
            fp_output = self._layer_forward(layer, x)

        # ---- 3. Compute quantization grid constants --------------------
        #   Δ (scale) and z (zero-point) are fixed throughout optimisation.
        w = layer.weight.data
        self._scale, self._zero_point = _compute_scale_zeropoint(
            w, self.n_bits, self.symmetric
        )
        scale = self._scale
        zp = self._zero_point

        # Shift weights onto the integer grid:
        #   w_scaled = w / Δ − z
        w_scaled = w / scale - zp
        self._weight_floor = torch.floor(w_scaled)   # shape == w.shape
        self._original_weight = w.clone()

        # ---- 4. Initialise V via inverse rectified-sigmoid  ------------
        #
        # We want h(V_init) ≈ frac(w_scaled) so the first step of the
        # optimisation starts near the RTN solution.
        #
        # h(V) = clip( σ(V)·(ζ−γ) + γ, 0, 1 )
        #
        # Inverting:  V_init = σ^{-1}( (frac − γ) / (ζ − γ) )
        #           = log(p / (1 − p))   where p = (frac − γ) / (ζ − γ)
        #
        # We clamp p to (ε, 1−ε) to avoid log(0).
        frac = (w_scaled - self._weight_floor).detach()   # fractional part ∈ [0, 1)
        p = (frac - _GAMMA) / (_ZETA - _GAMMA)
        p = p.clamp(1e-6, 1.0 - 1e-6)
        V_init = torch.log(p / (1.0 - p))  # inverse logistic = logit

        self._V = nn.Parameter(V_init.clone().to(device))

        # ---- 5. Build Adam optimiser over V only -----------------------
        optimizer = torch.optim.Adam([self._V], lr=self.learning_rate)

        # ---- 6. Optimisation loop  -------------------------------------
        logger.info(
            "AdaRound | layer=%s | n_bits=%d | steps=%d | λ=%.2e",
            type(layer).__name__, self.n_bits, n_steps, self.reg_lambda,
        )

        for step in range(n_steps):
            optimizer.zero_grad()

            # Linearly anneal β from beta_start → beta_end
            beta = self._anneal_beta(step, n_steps)

            # Compute quantized weight using current V
            w_hat = self._quantized_weight(w, scale, zp)

            # Temporarily swap layer weight to w_hat for forward pass
            layer.weight.data = w_hat
            q_output = self._layer_forward(layer, x)
            layer.weight.data = w   # restore immediately

            # ---- Reconstruction loss  ||W·x − W_hat(V)·x||_F^2 --------
            recon_loss = F.mse_loss(q_output, fp_output)

            # ---- Regulariser  R_β(V) = Σ 1 − |2·h(V_i) − 1|^β  --------
            #
            # When β is small this is a soft penalty (all V are penalised).
            # As β → ∞ only values near 0.5 are penalised, forcing h(V)→{0,1}.
            h = self._rectified_sigmoid(self._V)
            reg_loss = (1.0 - (2.0 * h - 1.0).abs().pow(beta)).sum()

            loss = recon_loss + self.reg_lambda * reg_loss

            loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0 or step == 0:
                logger.debug(
                    "  step %4d/%d | recon=%.4e | reg=%.4e | β=%.2f",
                    step + 1, n_steps,
                    recon_loss.item(), reg_loss.item(), beta,
                )

        # Restore the original FP32 weights into the layer
        layer.weight.data = w
        logger.info("AdaRound optimisation complete.")

    def get_rounded_weights(self) -> torch.Tensor:
        """Return the finalized integer-grid weights after optimisation.

        The returned tensor has the same shape as the original layer weight
        and lives on the same device.  Pass it directly to::

            layer.weight.data = opt.get_rounded_weights()

        before exporting or handing off to the MASE compiler.

        Returns
        -------
        torch.Tensor
            Dequantized weights  Δ · (floor(w/Δ) + h(V*))  on the original
            floating-point scale (NOT raw integers).  This is what you plug
            back into a standard PyTorch graph.
        """
        if self._V is None:
            raise RuntimeError(
                "Call optimize_layer() before get_rounded_weights()."
            )
        with torch.no_grad():
            return self._quantized_weight(
                self._original_weight, self._scale, self._zero_point
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rectified_sigmoid(V: torch.Tensor) -> torch.Tensor:
        """Compute h(V) = clip( σ(V)·(ζ−γ) + γ, 0, 1 ).

        This is the smooth "soft quantization" function from Eq.(7) in the
        paper.  It maps V ∈ ℝ → h ∈ [0, 1], and is differentiable everywhere.

          - When V → −∞ : σ(V) → 0, h → γ.clamp(0,1) = 0  (round down)
          - When V → +∞ : σ(V) → 1, h → ζ.clamp(0,1) = 1  (round up)
        """
        return torch.clamp(
            torch.sigmoid(V) * (_ZETA - _GAMMA) + _GAMMA,
            min=0.0,
            max=1.0,
        )

    def _quantized_weight(
        self,
        w: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute  w_hat = Δ · ( floor(w/Δ − z) + h(V) ) + z·Δ.

        This is the differentiable quantization operator: gradients flow
        through h(V) via the rectified sigmoid, not through floor().

        Parameters
        ----------
        w:
            Full-precision weight tensor.
        scale:
            Quantization step size Δ (scalar).
        zero_point:
            Zero-point offset z (scalar).

        Returns
        -------
        torch.Tensor
            Quantized-then-dequantized weight (same dtype/device as w).
        """
        # h(V) ∈ [0, 1]  — the learned rounding decision
        h = self._rectified_sigmoid(self._V)

        # floor(w/Δ − z) + h(V)  stays on the integer grid [w_floor, w_ceil]
        w_q = self._weight_floor + h

        # Clamp to representable integer range  [−2^(b-1), 2^(b-1)−1]  (symmetric)
        q_min = -(2 ** (self.n_bits - 1))
        q_max = 2 ** (self.n_bits - 1) - 1
        w_q = w_q.clamp(q_min, q_max)

        # Dequantize back to floating-point scale:  w_hat = Δ·(w_q + z)
        w_hat = scale * (w_q + zero_point)
        return w_hat

    def _anneal_beta(self, step: int, total_steps: int) -> float:
        """Linear annealing of β from beta_start to beta_end.

        β controls the sharpness of the regulariser R_β:
          - Small β  → soft penalty, allows V to explore freely early on.
          - Large β  → hard penalty, forces h(V) towards exactly 0 or 1.
        """
        progress = step / max(total_steps - 1, 1)  # ∈ [0, 1]
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    @staticmethod
    def _layer_forward(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through ``layer`` with ``no_grad`` for the
        FP32 reference, or with grad tracking when called inside the loop.

        We call this helper to centralise the forward dispatch so it works
        for both ``nn.Linear`` and ``nn.Conv2d`` without special-casing.
        """
        return layer(x)
