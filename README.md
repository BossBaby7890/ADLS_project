# HA-AdaRound: Final Quantization Enhancements

This branch contains the full combined enhancement set:

- Adaptive AdaRound scheduling
- Post-allocation bit-width refinement
- Calibration subset selection for profiling and AdaRound

## Main variants in this branch

### 1. Baseline
```bash
python scripts/run_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth \
    --adaround-steps 500 \
    --calib-batches 1
```

### 2. Adaptive AdaRound scheduling
```bash
python scripts/run_enhanced_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth \
    --adaround-steps 500 \
    --calib-batches 1
```

### 3. Bit-refined + adaptive AdaRound
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
```

### 4. Calibration-aware refined adaptive pipeline
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

## Profiling with calibration subset selection

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

## Expected output files

### Profiling

- `outputs/sensitivity_scores.json`
- `outputs/layer_sensitivity.json`
- `outputs/sensitivity_scores_random_256.json`
- `outputs/layer_sensitivity_random_256.json`
- `outputs/sensitivity_scores_class_balanced_256.json`
- `outputs/layer_sensitivity_class_balanced_256.json`

### Adaptive AdaRound

- `outputs/quant_config_enhanced.json`
- `outputs/checkpoints/checkpoint_adarounded_enhanced.pth`

### Refined adaptive AdaRound

- `outputs/quant_config_refined_enhanced.json`
- `outputs/bit_refinement_report.json`
- `outputs/checkpoints/checkpoint_refined_adarounded_enhanced.pth`

### Calibration-aware refined adaptive AdaRound

- `outputs/quant_config_calibrated_refined_enhanced.json`
- `outputs/calibration_refinement_report.json`
- `outputs/checkpoints/checkpoint_calibrated_refined_adarounded_enhanced.pth`

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

Outside MASE, the enhanced scripts fall back gracefully and save AdaRounded
checkpoints plus quantization configs for smoke-testing.
