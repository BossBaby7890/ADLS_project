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
| **Adaptive Scheduler** | `src/quantization/adaptive_scheduler.py` | Allocates layer-wise AdaRound optimisation budgets based on assigned bit-width and sensitivity score. More sensitive / lower-bit layers receive more reconstruction effort. |
| **Bit Refiner** | `src/refinement/bit_refiner.py` | Post-allocation refinement stage that upgrades a small number of highly sensitive 2-bit layers to 4-bit before AdaRound, improving robustness without replacing the baseline allocator. |
| **Calibration Selector** | `src/calibration/sample_selector.py` | Selects calibration subsets for profiling and AdaRound using strategies such as random and class-balanced sampling. Improves calibration quality without changing the base model or training loop. |
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

### 2. Run sensitivity profiling — Stage 1 (MASE terminal)

Profiles the fp32 model to produce per-layer sensitivity scores.

```bash
python scripts/run_profiling.py \
    --config     configs/base_config.yaml \
    --quant      configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth
```

Outputs:
- `outputs/layer_sensitivity.json` — module-level sensitivity scores
### 2b. Run sensitivity profiling with calibration subset selection

The profiling script also supports calibration subset selection strategies, allowing
sensitivity estimation to be computed from a smaller and more controlled calibration set.

#### Random calibration subset

```bash
python scripts/run_profiling.py \
    --config configs/base_config.yaml \
    --quant  configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth \
    --calib-strategy random \
    --calib-samples 256

Class-balanced calibration sheet
python scripts/run_profiling.py \
    --config configs/base_config.yaml \
    --quant  configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth \
    --calib-strategy class_balanced \
    --calib-samples 256

### 3. Run AdaRound with mixed-precision allocation — Stages 2 / 2.5 / 3 (MASE terminal)

Assigns bit-widths from sensitivity scores (Stage 2), runs layer-wise adaptive
rounding over a calibration batch (Stage 2.5), generates the CHOP config and
applies `quantize_transform_pass` to produce the final quantized model (Stage 3).

```bash
python scripts/run_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth
```

**AdaRound options:**

| Flag | Default | Effect |
|---|---|---|
| `--adaround-steps` | `500` | Optimisation steps per layer (paper: 10 000) |
| `--calib-batches` | `1` | Calibration batches used per layer |
| `--skip-adaround` | off | Skip to RTN rounding (faster, lower accuracy) |

Outputs:
- `outputs/quant_config.json` — CHOP quantization pass config
- `outputs/checkpoints/checkpoint_quantized.pth` — AdaRounded + MASE-quantized weights

### 3b. Run adaptive AdaRound scheduling (enhanced Stage 2.5 / 2.6)

This variant keeps the original bit-width allocation stage unchanged, but allocates
different AdaRound optimisation budgets to different layers based on:

- assigned bit-width
- sensitivity score

Lower-bit and more sensitive layers receive more reconstruction effort.

```bash
python scripts/run_enhanced_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth \
    --adaround-steps 500 \
    --calib-batches 1

### 3c. Run bit-refined + adaptive AdaRound pipeline (enhanced Stage 2 / 2.2 / 2.5 / 2.6 / 3)

This variant adds a lightweight post-allocation refinement stage before AdaRound.
Starting from the threshold-based mixed-precision bit-map, it upgrades a small number
of highly sensitive 2-bit layers to 4-bit, then runs adaptive AdaRound scheduling.

```bash
python scripts/run_refined_enhanced_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth \
    --adaround-steps 500 \
    --calib-batches 1 \
    --max-rescues 3 \
    --min-rescue-sensitivity 0.05

### 3d. Run calibration-aware refined adaptive pipeline

This is the most complete enhanced pipeline currently implemented. It combines:

- calibration subset selection
- post-allocation bit-width refinement
- adaptive AdaRound scheduling

```bash
python scripts/run_calibrated_refined_enhanced_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity_class_balanced_256.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth \
    --adaround-steps 500 \
    --calib-batches 1 \
    --max-rescues 3 \
    --min-rescue-sensitivity 0.05 \
    --calib-strategy class_balanced \
    --calib-samples 256


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

Outputs:
- `outputs/model_quantized.onnx` — quantized graph ready for MASE HLS / TVM

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

```md
### Checkpoint requirement
For meaningful quantization experiments, the pipeline expects a pretrained fp32 checkpoint at:

## Which script should I run?

The repository now supports four main execution variants.

### 1. Baseline HA-AdaRound pipeline
Use the original script when you want the unmodified baseline pipeline:

```bash
python scripts/run_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth \
    --adaround-steps 500 \
    --calib-batches 1

## Expected output files

Depending on which script is used, the repository may generate some or all of the following files.

### Sensitivity profiling outputs
Produced by `scripts/run_profiling.py`:

- `outputs/sensitivity_scores.json`
- `outputs/layer_sensitivity.json`

Optional saved variants:
- `outputs/sensitivity_scores_random_256.json`
- `outputs/layer_sensitivity_random_256.json`
- `outputs/sensitivity_scores_class_balanced_256.json`
- `outputs/layer_sensitivity_class_balanced_256.json`

### Enhanced quantization outputs

#### Adaptive AdaRound
- `outputs/quant_config_enhanced.json`
- `outputs/checkpoints/checkpoint_adarounded_enhanced.pth`

#### Refined adaptive AdaRound
- `outputs/quant_config_refined_enhanced.json`
- `outputs/bit_refinement_report.json`
- `outputs/checkpoints/checkpoint_refined_adarounded_enhanced.pth`

#### Calibration-aware refined adaptive AdaRound
- `outputs/quant_config_calibrated_refined_enhanced.json`
- `outputs/calibration_refinement_report.json`
- `outputs/checkpoints/checkpoint_calibrated_refined_adarounded_enhanced.pth`

### MASE/CHOP-enabled outputs
When the scripts are executed inside a MASE/CHOP-enabled environment, they may additionally produce final compiler-compatible quantized checkpoints after applying `quantize_transform_pass`.

## Environment Notes

### Checkpoint requirement
For meaningful quantization experiments, the pipeline expects a pretrained fp32 checkpoint at:

```text
outputs/checkpoints/checkpoint_best.pth



## References

- **HAWQ**: Dong et al., *"HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision"*, ICCV 2019.
- **HAWQ-V2**: Dong et al., *"HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks"*, NeurIPS 2020.
- **APQ**: Wang et al., *"APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"*, CVPR 2020.
- **AdaRound**: Nagel et al., *"Up or Down? Adaptive Rounding for Post-Training Quantization"*, ICML 2020.
- **MASE**: *Machine-Learning Accelerator System Exploration* framework — [github.com/DeepWok/mase](https://github.com/DeepWok/mase).
