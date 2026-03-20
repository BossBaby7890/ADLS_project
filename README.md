# APQ-Lite: 1st-Order Gradient-Aware Mixed-Precision Quantization

A lightweight, research-grade implementation of **automated mixed-precision
quantization (APQ)** using first-order gradient magnitudes as a cheap proxy
for the Hessian.  The resulting per-layer bit-width assignments are compiled
directly into the **MASE/CHOP** hardware compiler for FPGA/ASIC deployment.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APQ-Lite Pipeline                               │
│                                                                         │
│  ┌──────────────┐   JSON    ┌──────────────┐   BitMap  ┌─────────────┐ │
│  │  Stage 1     │ ────────► │  Stage 2     │ ────────► │  Stage 3    │ │
│  │  Profiler    │           │  Allocator   │           │  Compiler   │ │
│  │              │           │              │           │             │ │
│  │ Gradient     │           │ Threshold-   │           │ MASE/CHOP   │ │
│  │ sensitivity  │           │ based        │           │ config      │ │
│  │ (||∇W||²)   │           │ bit-width    │           │ generation  │ │
│  │              │           │ assignment   │           │             │ │
│  └──────────────┘           └──────────────┘           └──────┬──────┘ │
│         ▲                                                      │        │
│         │                                              CHOP    │        │
│  ┌──────┴──────┐                                   pass cfg   │        │
│  │  Models     │◄──────────────────────────────────────────────┘        │
│  │  ResNet20   │                                                        │
│  │  MobileNet  │   ┌──────────────┐        ┌──────────────────────────┐ │
│  │  …          │──►│  Stage 4     │──────► │  Stage 5                 │ │
│  └─────────────┘   │  Engine      │        │  ONNX Export             │ │
│                    │              │        │  (MASE HLS / TVM target) │ │
│                    │  fp32 train  │        └──────────────────────────┘ │
│                    │  + QAT loop  │                                      │
│                    └──────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────┘

          MASE Terminal  ◄──────────────────────  VS Code
          (profiling,                            (analysis,
           QAT, export)                          notebooks)
```

### Module Descriptions

| Module | File | Role |
|---|---|---|
| **Profiler** | `src/profiler/sensitivity.py` | Accumulates `‖∇W‖²` over a calibration set to score each layer's sensitivity to quantization. Acts as a 1st-order Hessian proxy (HAWQ-lite). |
| **Allocator** | `src/allocator/bit_mapper.py` | Applies threshold policy to sensitivity scores to assign `{2, 4, 8}`-bit widths per layer. Respects first/last layer overrides and hardware budget. |
| **Compiler** | `src/compiler/mase_integration.py` | Translates the bit-map into a MASE/CHOP `quantization_config` dict (weight/activation widths + fractional bits). Wraps it for `quantize_transform_pass`. |
| **Models** | `src/models/` | Registerable architecture zoo (ResNet-20/32/56). Consistent naming enables the profiler and allocator to match layers across stages. |
| **Engine** | `src/engine/trainer.py` | Unified fp32 pre-training and QAT fine-tuning loop with AMP, grad-clip, and checkpoint management. |
| **Engine** | `src/engine/evaluator.py` | Top-1/5 accuracy, cross-entropy loss, throughput, and latency benchmarking. Serialises results for notebook analysis. |

---

## Repository Structure

```
APQ-Lite/
├── configs/
│   ├── base_config.yaml        # Global hyperparameters
│   └── quant_params.yaml       # Bit-width thresholds & MASE keys
│
├── src/
│   ├── profiler/
│   │   └── sensitivity.py      # Stage 1 — gradient norm profiling
│   ├── allocator/
│   │   └── bit_mapper.py       # Stage 2 — sensitivity → bit-width
│   ├── compiler/
│   │   └── mase_integration.py # Stage 3 — CHOP config generation
│   ├── models/
│   │   ├── __init__.py         # Model registry
│   │   └── resnet.py           # ResNet-20/32/56 for CIFAR
│   └── engine/
│       ├── trainer.py          # fp32 + QAT training loop
│       └── evaluator.py        # Accuracy & latency evaluation
│
├── scripts/                    # Run from MASE terminal
│   ├── run_profiling.py        # Stage 1 entry-point
│   ├── run_qat.py              # Stage 2 + 3 + QAT entry-point
│   └── export_onnx.py          # ONNX export entry-point
│
├── notebooks/
│   └── Results_Visualization.ipynb  # VS Code analysis
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pre-train the fp32 baseline (MASE terminal)

```bash
python scripts/run_profiling.py --config configs/base_config.yaml
```

> **Note:** Load a pretrained checkpoint with `--checkpoint` to skip training.

### 3. Run QAT with mixed-precision allocation (MASE terminal)

```bash
python scripts/run_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth
```

### 4. Export to ONNX (MASE terminal)

```bash
python scripts/export_onnx.py \
    --config     configs/base_config.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth \
    --output     outputs/model_quantized.onnx
```

### 5. Analyse results (VS Code)

Open `notebooks/Results_Visualization.ipynb` to compare baseline vs.
quantized accuracy, visualise per-layer bit-width assignments, and plot
the sensitivity score distribution.

---

## Configuration

All behaviour is controlled via two YAML files:

- **`configs/base_config.yaml`** — dataset, model, training hyperparameters,
  profiling settings, QAT epochs, logging.
- **`configs/quant_params.yaml`** — sensitivity thresholds, available
  bit-widths, layer overrides, hardware budget, MASE key names.

---

## References

- **HAWQ**: Dong et al., *"HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision"*, ICCV 2019.
- **HAWQ-V2**: Dong et al., *"HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks"*, NeurIPS 2020.
- **APQ**: Wang et al., *"APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"*, CVPR 2020.
- **MASE**: *Machine-Learning Accelerator System Exploration* framework — [github.com/DeepWok/mase](https://github.com/DeepWok/mase).