# HA-AdaRound: Calibration-Aware Refined Quantization Pipeline

This branch contains:

- Adaptive AdaRound scheduling
- Post-allocation bit-width refinement
- Calibration subset selection for profiling and AdaRound

## What is in this branch

This branch extends the pipeline with calibration-aware profiling and a calibration-aware refined adaptive quantization pipeline.

## Main files

- `src/quantization/adaptive_scheduler.py`
- `src/refinement/bit_refiner.py`
- `src/calibration/sample_selector.py`
- `scripts/run_profiling.py`
- `scripts/run_calibrated_refined_enhanced_qat.py`

## Run sensitivity profiling
```bash
python scripts/run_profiling.py \
    --config configs/base_config.yaml \
    --quant  configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth
```

Outputs:

- `outputs/sensitivity_scores.json`
- `outputs/layer_sensitivity.json`

## Run sensitivity profiling with calibration subset selection

### Random subset
```bash
python scripts/run_profiling.py \
    --config configs/base_config.yaml \
    --quant  configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth \
    --calib-strategy random \
    --calib-samples 256
```

### Class-balanced subset
```bash
python scripts/run_profiling.py \
    --config configs/base_config.yaml \
    --quant  configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth \
    --calib-strategy class_balanced \
    --calib-samples 256
```

Example saved outputs:

- `outputs/layer_sensitivity_random_256.json`
- `outputs/layer_sensitivity_class_balanced_256.json`
- `outputs/sensitivity_scores_random_256.json`
- `outputs/sensitivity_scores_class_balanced_256.json`

## Run calibration-aware refined adaptive pipeline
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
```

Outputs:

- `outputs/quant_config_calibrated_refined_enhanced.json`
- `outputs/calibration_refinement_report.json`
- `outputs/checkpoints/checkpoint_calibrated_refined_adarounded_enhanced.pth`

If executed inside a MASE/CHOP-enabled environment, the script also applies
`quantize_transform_pass` and saves the final quantized checkpoint.
Outside MASE, it falls back gracefully and saves the AdaRounded checkpoint plus
quantization config only.

## Environment notes

### Checkpoint requirement

Meaningful runs require:
```
outputs/checkpoints/checkpoint_best.pth
```

This checkpoint is not included in the repository.

### MASE / CHOP dependency

Full compiler-stage execution requires:

- `chop`
- `quantize_transform_pass`
- `MaseGraph`

Without CHOP, the script still runs the adaptive AdaRound pipeline and saves
intermediate outputs for smoke-testing.
