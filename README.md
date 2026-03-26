# HA-AdaRound: Hessian-Aware Adaptive Rounding for Mixed-Precision Quantization

A lightweight, research-grade implementation of **Hessian-Aware AdaRound (HA-AdaRound)** —
automated mixed-precision quantization that combines first-order gradient sensitivity
profiling with layer-wise adaptive rounding, compiled directly into the **MASE/CHOP**
hardware compiler for FPGA/ASIC deployment.

The pipeline avoids full Quantization-Aware Training (QAT). Instead, it uses AdaRound
to perform layer-wise reconstruction loss minimisation over a small calibration set,
recovering accuracy at low bit-widths without a fine-tuning training loop.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        HA-AdaRound Pipeline                                  │
│                                                                              │
│  ┌─────────────┐  JSON   ┌─────────────┐  BitMap  ┌──────────────────────┐  │
│  │  Stage 1    │ ──────► │  Stage 2    │ ───────► │  Stage 2.5           │  │
│  │  Profiler   │         │  Allocator  │          │  AdaRound            │  │
│  │             │         │             │          │                      │  │
│  │ E[||∇W||²]  │         │ Threshold   │          │ Layer-wise learned   │  │
│  │ per layer   │         │ → {2,4,8}   │          │ rounding via V       │  │
│  │ (Hessian    │         │ -bit/layer  │          │ min ||Wx-Ŵ(V)x||²_F  │  │
│  │  proxy)     │         │             │          │ + β-annealing reg.   │  │
│  └─────────────┘         └─────────────┘          └──────────┬───────────┘  │
│         ▲                                                     │              │
│         │                                          rounded    │              │
│  ┌──────┴──────┐                                  weights     ▼              │
│  │  Models     │                         ┌──────────────────────────────┐   │
│  │  ResNet20   │                         │  Stage 3                     │   │
│  │  ResNet32   │                         │  MASE/CHOP Compiler          │   │
│  │  ResNet56   │                         │                              │   │
│  └──────┬──────┘                         │  MaseConfigGenerator →       │   │
│         │                                │  quantize_transform_pass     │   │
│         │                                │  → checkpoint_quantized.pth  │   │
│         │                                │  → quant_config.json         │   │
│         │                                └──────────────┬───────────────┘   │
│         │                                               │                   │
│         └───────────────────────────────────────────────┤                   │
│                                                         ▼                   │
│                                              ┌──────────────────────┐       │
│                                              │  Stage 4             │       │
│                                              │  ONNX Export         │       │
│                                              │  (MASE HLS / TVM)    │       │
│                                              └──────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────────┘

          MASE Terminal  ◄──────────────────────────  VS Code
          (profiling,                                (analysis,
           AdaRound,                                 notebooks)
           export)
```

### Module Descriptions

| Module | File | Role |
|---|---|---|
| **Profiler** | `src/profiler/sensitivity.py` | Accumulates `E[‖∇W‖²]` over a calibration set to score each layer's sensitivity to quantization. Acts as a 1st-order Hessian proxy (HAWQ-lite). |
| **Allocator** | `src/allocator/bit_mapper.py` | Applies threshold policy to sensitivity scores to assign `{2, 4, 8}`-bit widths per layer. Respects first/last-layer overrides and hardware budget. |
| **AdaRound** | `src/quantization/adaround.py` | Layer-wise adaptive rounding. For each layer, optimises continuous rounding variables `V` to minimise `‖Wx − Ŵ(V)x‖²_F + λ·R_β(V)` over a small calibration batch. Replaces QAT. |
| **Compiler** | `src/compiler/mase_integration.py` | Translates the bit-map into a MASE/CHOP `quantization_config` dict (weight/activation widths + fractional bits). Wraps it for `quantize_transform_pass`. |
| **Models** | `src/models/` | Registerable architecture zoo (ResNet-20/32/56). Consistent naming enables the profiler and allocator to match layers across stages. |
| **Evaluator** | `src/engine/evaluator.py` | Top-1/5 accuracy, cross-entropy loss, throughput, and latency benchmarking. Serialises results for notebook analysis. |

---

## Repository Structure

```
HA-AdaRound/
├── configs/
│   ├── base_config.yaml        # Global hyperparameters (data, model, profiling, eval)
│   └── quant_params.yaml       # Bit-width thresholds, layer overrides & MASE keys
│
├── src/
│   ├── profiler/
│   │   └── sensitivity.py      # Stage 1 — gradient norm profiling
│   ├── allocator/
│   │   └── bit_mapper.py       # Stage 2 — sensitivity → bit-width assignment
│   ├── quantization/
│   │   └── adaround.py         # Stage 2.5 — adaptive rounding optimiser
│   ├── compiler/
│   │   └── mase_integration.py # Stage 3 — CHOP config generation
│   ├── models/
│   │   ├── __init__.py         # Model registry + build_model()
│   │   └── resnet.py           # ResNet-20/32/56 for CIFAR-10/100
│   └── engine/
│       └── evaluator.py        # Accuracy & latency evaluation
│
├── scripts/                    # Run from MASE terminal
│   ├── run_profiling.py        # Stage 1 entry-point
│   ├── run_qat.py              # Stage 2 + 2.5 + 3 entry-point
│   └── export_onnx.py          # Stage 4 — ONNX export entry-point
│
├── notebooks/
│   └── Results_Visualization.ipynb  # VS Code analysis
│
├── outputs/                    # Generated at runtime (gitignored)
│   ├── sensitivity_scores.json      # Raw per-parameter scores (Stage 1)
│   ├── layer_sensitivity.json       # Module-level scores (Stage 1)
│   ├── quant_config.json            # CHOP pass config (Stage 3)
│   └── checkpoints/
│       ├── checkpoint_best.pth      # fp32 pretrained weights (Stage 1 input)
│       └── checkpoint_quantized.pth # AdaRounded + MASE-quantized (Stage 4 input)
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

### 2. Run **Stage 1**: **Joint Sensitivity Profiling** (MASE terminal)
Profiles fp32 model → **grad_norm + Hessian trace + Taylor scores** (new!)
```bash
python scripts/run_profiling.py \
    --config     configs/base_config.yaml \
    --quant      configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth
```
**Outputs**: `outputs/layer_sensitivity.json` (nested metrics)

### 3. Run **Stage 1.5-3.5**: **Automated Policy Search + Pruning + Quantization** (MASE terminal)
**NEW**: Greedy/CMA-ES search finds optimal **pruning+bitwidth policy** → Taylor pruning → KD recovery → AdaRound → MASE compile → resource parsing
```bash
# Fast test (greedy only)
python scripts/run_search.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth \
    --search-strategy greedy

# Full run (greedy + CMA-ES, best wins)
python scripts/run_search.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth
```
**Outputs**: 
- `outputs/best_policy.json` (winning strategy)
- `outputs/checkpoints/checkpoint_pruned.pth`
- `outputs/quant_config.json`
- `outputs/resource_summary.json` (LUT/DSP estimates)

### 4. Export to ONNX — Stage 4 (MASE terminal)

Reloads the quantized checkpoint, re-applies the CHOP pass to restore the
quantized graph structure, then exports to ONNX.

```bash
python scripts/export_onnx.py \
    --config     configs/base_config.yaml \
    --quant      configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_quantized.pth \
    --output     outputs/model_quantized.onnx
```
**Output**: `outputs/model_quantized.onnx` (FPGA/TVM ready)


### 5. Analyse results (VS Code)

Open `notebooks/Results_Visualization.ipynb` to compare baseline vs. quantized
accuracy, visualise per-layer bit-width assignments, and plot the sensitivity
score distribution.

---

## Configuration

All behaviour is controlled via two YAML files:

**`configs/base_config.yaml`** — dataset path, model architecture, fp32 training
hyperparameters, profiling settings, evaluation batch size, logging.

**`configs/quant_params.yaml`** — sensitivity thresholds for `{2, 4, 8}`-bit
assignment, first/last-layer overrides, hardware budget targets, MASE/CHOP
compiler key names.

Key sections:

```yaml
# base_config.yaml
data:
  batch_size: 128          # dataloader batch size (used for AdaRound calibration)

profiling:
  num_batches: 50          # calibration batches for gradient norm accumulation
```

```yaml
# quant_params.yaml
thresholds:
  high: 0.70               # score >= high  →  8-bit
  mid:  0.35               # score >= mid   →  4-bit  (else 2-bit)

layer_overrides:
  first_layer_bits: 8      # always 8-bit for stem
  last_layer_bits: 8       # always 8-bit for classifier head
```

---

## References

- **HAWQ**: Dong et al., *"HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision"*, ICCV 2019.
- **HAWQ-V2**: Dong et al., *"HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks"*, NeurIPS 2020.
- **APQ**: Wang et al., *"APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"*, CVPR 2020.
- **AdaRound**: Nagel et al., *"Up or Down? Adaptive Rounding for Post-Training Quantization"*, ICML 2020.
- **MASE**: *Machine-Learning Accelerator System Exploration* framework — [github.com/DeepWok/mase](https://github.com/DeepWok/mase).
