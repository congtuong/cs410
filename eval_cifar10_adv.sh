#!/bin/bash
#
# Evaluation script for CIFAR-10 Adversarial Training for Free
# 

# Default parameters
DATA_PATH="/mlcv2/WorkingSpace/Personal/tuongbck/cs410/cifar10/cifar-10-batches-py"
BATCH_SIZE=128
EPOCHS=200
LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=0.0001
N_REPEATS=4
FGSM_STEP=2
CLIP_EPS=8
PRINT_FREQ=50
SAVE_FREQ=5
MODEL="wrn_32_10"
CHECKPOINT=""

# Parse command-line options
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --momentum)
            MOMENTUM="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --n-repeats)
            N_REPEATS="$2"
            shift 2
            ;;
        --fgsm-step)
            FGSM_STEP="$2"
            shift 2
            ;;
        --clip-eps)
            CLIP_EPS="$2"
            shift 2
            ;;
        --print-freq)
            PRINT_FREQ="$2"
            shift 2
            ;;
        --save-freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --data-path PATH      Path to CIFAR-10 data (default: $DATA_PATH)"
            echo "  --batch-size SIZE     Mini-batch size (default: $BATCH_SIZE)"
            echo "  --epochs NUM          Number of total epochs to run (default: $EPOCHS)"
            echo "  --lr RATE             Initial learning rate (default: $LR)"
            echo "  --momentum VAL        Momentum (default: $MOMENTUM)"
            echo "  --weight-decay VAL    Weight decay (default: $WEIGHT_DECAY)"
            echo "  --n-repeats NUM       Number of repeats for free adversarial training (default: $N_REPEATS)"
            echo "  --fgsm-step SIZE      Step size for FGSM (default: $FGSM_STEP)"
            echo "  --clip-eps SIZE       Maximum perturbation size (L∞ norm) (default: $CLIP_EPS)"
            echo "  --print-freq NUM      Print frequency (default: $PRINT_FREQ)"
            echo "  --save-freq NUM       Save frequency (default: $SAVE_FREQ)"
            echo "  --model NAME          Model architecture: resnet18, resnet34, resnet50, wrn_32_10 (default: $MODEL)"
            echo "  --checkpoint PATH     Path to checkpoint file for evaluation (required)"
            echo "  --help                Display this help message and exit"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage"
            exit 1
            ;;
    esac
done

# Check if checkpoint is provided
if [[ -z "$CHECKPOINT" ]]; then
    echo "Error: --checkpoint is required for evaluation"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# Check if checkpoint file exists
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Error: Checkpoint file '$CHECKPOINT' does not exist"
    exit 1
fi

# Create output directory based on configuration
EXPERIMENT_NAME="${MODEL}_eps${CLIP_EPS}_step${FGSM_STEP}_rep${N_REPEATS}"

OUTPUT_DIR="experiments_cifar10/${EXPERIMENT_NAME}"
mkdir -p "$OUTPUT_DIR"

# Log evaluation configuration
echo "=== CIFAR-10 Adversarial Evaluation Configuration ==="
echo "Data path: $DATA_PATH"
echo "Model: $MODEL"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LR"
echo "Momentum: $MOMENTUM"
echo "Weight decay: $WEIGHT_DECAY"
echo "Adversarial repeats: $N_REPEATS"
echo "FGSM step size: $FGSM_STEP"
echo "Perturbation limit (eps): $CLIP_EPS"
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "=================================================="

# Run the evaluation
python cifar10_adv_training.py \
    --data-path "$DATA_PATH" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --momentum "$MOMENTUM" \
    --weight-decay "$WEIGHT_DECAY" \
    --n-repeats "$N_REPEATS" \
    --fgsm-step "$FGSM_STEP" \
    --clip-eps "$CLIP_EPS" \
    --print-freq "$PRINT_FREQ" \
    --save-freq "$SAVE_FREQ" \
    --model "$MODEL" \
    --evaluate \
    --checkpoint "$CHECKPOINT" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

echo "Evaluation completed! Results saved to $OUTPUT_DIR"
