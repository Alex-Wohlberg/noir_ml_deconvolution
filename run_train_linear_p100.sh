#!/usr/bin/env bash
# Train the conditional DDPM on the anchor-100 linear dataset, asinh-domain loss.
#
#   ./run_train_linear_p100.sh              # 15 epochs (the gated short run)
#   EPOCHS=100 ./run_train_linear_p100.sh   # full run
#   SANITY=1 ./run_train_linear_p100.sh     # --overfit-one-batch instead
#
# Needs a GPU: on CPU a batch-8 step takes ~3.3 s. Run it on Vista, with
# PYTHON pointed at the env that has torch there and DATA_DIR at the copied
# dataset:
#   PYTHON=python DATA_DIR=/path/on/vista/m31bK50_linear_p100 \
#       EPOCHS=100 ./run_train_linear_p100.sh
set -euo pipefail

PYTHON="${PYTHON:-/opt/conda/miniconda3/envs/py313/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/home/alex/noir_ml/global/ml-decon/data/m31bK50_linear_p100}"
CKPT_DIR="${CKPT_DIR:-$HERE/checkpoints_cond_diffusion_linear_p100}"
EPOCHS="${EPOCHS:-15}"

if [[ "${SANITY:-0}" == "1" ]]; then
    # Memorize a single batch. Expect the fixed-eval loss to fall below 0.05.
    # It reaches PASS quickly at this anchor because nearly all the signal is
    # below the noise floor, which makes eps-prediction easy -- a PASS here is
    # a plumbing check, not evidence the model learned the field.
    exec "$PYTHON" -u "$HERE/train_conditional_diffusion_linear.py" \
        --data-dir "$DATA_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --overfit-one-batch --sanity-steps 3000
fi

# Defaults left as shipped on purpose: read the printed
# "train X (eps Y stretch Z)" breakdown before touching the weights. If stretch
# stays flat while eps falls, the asinh term is not doing work and the ratio
# needs raising -- do not guess it up front.
exec "$PYTHON" -u "$HERE/train_conditional_diffusion_linear.py" \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CKPT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size 32 \
    --lr 1e-4 \
    --eps-weight 1.0 \
    --stretch-weight 1.0 \
    --p-uncond 0.15 \
    --identity-frac 0.15 \
    --low-t-frac 0.25 \
    --num-workers 4
