#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_model_dir> <output_dir> [profile: 4b|3.5b|3b] [residual: bf16|q8]"
    echo ""
    echo "Examples:"
    echo "  # Official Hugging Face BF16 -> TFQ3 4b with pure BF16 base (~14.37 GiB, zero calibration)"
    echo "  $0 ./Qwen3.8-27B ./Qwen3.8-27B-TFQ3 4b bf16"
    echo ""
    echo "  # MLX 8-bit/64 -> TFQ3 4b with 8-bit residual base (~13.23 GiB, -1.75 GiB vs MLX-4bit)"
    echo "  $0 ./Qwen3.8-27B-MLX-8bit ./Qwen3.8-27B-TFQ3-Q8 4b q8"
    exit 1
fi

SRC="$1"
OUT="$2"
PROFILE="${3:-4b}"
RESIDUAL="${4:-bf16}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

if [ ! -x "$BUILD/tfq3-qwen27-convert" ] || [ ! -x "$BUILD/tfq3-qwen27-package" ]; then
    echo "Building TFQ3 converters..."
    make -C "$ROOT" qwen27 tools
fi

echo "=== [1/3] Quantizing dense weights (profile=${PROFILE}, 100% data-free) ==="
"$BUILD/tfq3-qwen27-convert" \
    --model "$SRC" \
    --out "$OUT" \
    --profile "$PROFILE" \
    --head-bits 4

echo "=== [2/3] Packaging residual weights, config.json, and tokenizer (residual=${RESIDUAL}) ==="
"$BUILD/tfq3-qwen27-package" \
    --src "$SRC" \
    --out "$OUT" \
    --residual "$RESIDUAL"

echo "=== [3/3] Validating packaged TFQ3 checkpoint ==="
"$BUILD/tfq3-inspect" "$OUT"
echo "Done: $OUT"
