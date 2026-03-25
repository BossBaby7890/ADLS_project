"""src.engine — Evaluation loop for APQ-Lite experiments."""

# QAT removed — Trainer removed with it; AdaRound handles post-quantization error correction
from .evaluator import Evaluator

__all__ = ["Evaluator"]
