"""
APQ-Lite: 1st-Order Gradient-Aware Mixed-Precision Quantization
================================================================
Top-level package for the APQ-Lite research framework.

Pipeline overview
-----------------
1. Profiler   — compute per-layer sensitivity via squared gradient norms.
2. Allocator  — map sensitivity scores to {2, 4, 8}-bit width assignments.
3. Compiler   — emit a quantization_config dict for the MASE/CHOP compiler.
4. Engine     — run standard training, QAT, and evaluation loops.
5. Models     — architecture definitions (ResNet, MobileNetV2, …).
"""

__version__ = "0.1.0"
__author__ = "APQ-Lite Research Team"
