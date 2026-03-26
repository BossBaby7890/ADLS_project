# HA-AdaRound: Bit Refinement + Adaptive AdaRound

This branch contains:

- Adaptive AdaRound scheduling
- Post-allocation bit-width refinement

## What is in this branch

This branch improves mixed-precision robustness without changing the model architecture or requiring full retraining.

The bit refinement stage upgrades a small number of highly sensitive 2-bit layers to 4-bit before AdaRound.

## Main files

- `src/quantization/adaptive_scheduler.py`
- `src/refinement/bit_refiner.py`
- `scripts/run_enhanced_qat.py`
- `scripts/run_refined_enhanced_qat.py`

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

## Run refined + adaptive AdaRound
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

Outputs:

- `outputs/quant_config_refined_enhanced.json`
- `outputs/bit_refinement_report.json`
- `outputs/checkpoints/checkpoint_refined_adarounded_enhanced.pth`

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
