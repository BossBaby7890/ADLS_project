# HA-AdaRound: Adaptive AdaRound Scheduling for Mixed-Precision Quantization

This branch contains the adaptive AdaRound scheduling enhancement only.

## What is in this branch

This branch keeps the original threshold-based bit-width allocation unchanged, and adds:

- Adaptive layer-wise AdaRound step scheduling
- Adaptive calibration effort per layer

Lower-bit and more sensitive layers receive more reconstruction effort.

## Main files

- `src/quantization/adaptive_scheduler.py`
- `scripts/run_enhanced_qat.py`

## Run profiling
```bash
python scripts/run_profiling.py \
    --config configs/base_config.yaml \
    --quant  configs/quant_params.yaml \
    --checkpoint outputs/checkpoints/checkpoint_best.pth
```

Outputs:

- `outputs/sensitivity_scores.json`
- `outputs/layer_sensitivity.json`

## Run adaptive AdaRound scheduling
```bash
python scripts/run_enhanced_qat.py \
    --config      configs/base_config.yaml \
    --quant       configs/quant_params.yaml \
    --sensitivity outputs/layer_sensitivity.json \
    --pretrained  outputs/checkpoints/checkpoint_best.pth \
    --adaround-steps 500 \
    --calib-batches 1
```

Outputs:

- `outputs/quant_config_enhanced.json`
- `outputs/checkpoints/checkpoint_adarounded_enhanced.pth`

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
