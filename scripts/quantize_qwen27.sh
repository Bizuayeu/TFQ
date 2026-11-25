#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <source_model_dir> <output_dir> [profile: 4b|3.5b|3b] [residual: bf16|q8]"
    echo ""
    echo "Works on both Apple Silicon (Metal) and Linux + NVIDIA GPU (CUDA)."
    echo "Examples:"
    echo "  $0 ./Qwen3.8-27B ./Qwen3.8-27B-TFQ3 4b bf16"
    echo "  $0 ./Qwen3.8-27B-MLX-8bit ./Qwen3.8-27B-TFQ3-Q8 4b q8"
    exit 1
fi

SRC="$1"
OUT="$2"
PROFILE="${3:-4b}"
RESIDUAL="${4:-bf16}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"

if [ "$(uname -s)" = "Darwin" ]; then
    CONV="$BUILD/tfq3-qwen27-convert"
    if [ ! -x "$CONV" ] || [ ! -x "$BUILD/tfq3-qwen27-package" ]; then
        make -C "$ROOT" tools qwen27
    fi
else
    CONV="$BUILD/tfq3-qwen27-convert-cuda"
    if [ ! -x "$CONV" ] || [ ! -x "$BUILD/tfq3-qwen27-package" ]; then
        make -C "$ROOT" cuda
    fi
fi

echo "=== [1/3] Quantizing dense weights (profile=${PROFILE}, 100% data-free) ==="
"$CONV" \
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
