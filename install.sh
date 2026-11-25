#!/usr/bin/env bash
set -euo pipefail

# One-command installer for tfq3 (Apple Silicon Metal & NVIDIA CUDA)
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ashhart/TFQ/main/install.sh | bash
#
# Optional environment variables:
#   PREFIX        Install prefix (default: $HOME/.local -> installs $PREFIX/bin/tfq3)
#   CUDA_ARCH     CUDA target arch on Linux (default: auto-detected via nvidia-smi or 'native')
#   TFQ3_SRC_DIR  Source checkout path (default: $HOME/.local/share/tfq3/src)
#   TFQ3_REF      Git branch or tag to install (default: main)

REPO="${TFQ3_REPO:-https://github.com/ashhart/TFQ.git}"
REF="${TFQ3_REF:-main}"
PREFIX="${PREFIX:-$HOME/.local}"
SRC_DIR="${TFQ3_SRC_DIR:-$HOME/.local/share/tfq3/src}"

die() {
    echo "install.sh: error: $*" >&2
    exit 1
}

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required tool '$1'"
}

need_cmd git
need_cmd make
need_cmd python3

OS="$(uname -s)"
MAKE_ARGS=()

if [ "$OS" = "Darwin" ]; then
    [ "$(uname -m)" = "arm64" ] || die "macOS builds require Apple Silicon (arm64)"
    xcrun -sdk macosx -find metal >/dev/null 2>&1 || die "Xcode Metal toolchain not found (install Xcode and run: xcodebuild -downloadComponent MetalToolchain)"
    echo "==> Detected macOS Apple Silicon (Metal)"
elif [ "$OS" = "Linux" ]; then
    NVCC_BIN="${NVCC:-$(command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)}"
    [ -x "$NVCC_BIN" ] || die "CUDA compiler 'nvcc' not found (install CUDA Toolkit 12+/13+ or set NVCC=/path/to/nvcc)"
    need_cmd g++
    if [ -z "${CUDA_ARCH:-}" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .\r\n')"
            if [[ "$CC" =~ ^[0-9]{2,3}$ ]]; then
                CUDA_ARCH="sm_${CC}"
            else
                CUDA_ARCH="native"
            fi
        else
            CUDA_ARCH="native"
        fi
    fi
    echo "==> Detected Linux NVIDIA CUDA (NVCC=${NVCC_BIN}, CUDA_ARCH=${CUDA_ARCH})"
    MAKE_ARGS+=(NVCC="$NVCC_BIN" CUDA_ARCH="$CUDA_ARCH")
else
    die "unsupported OS '${OS}' (TFQ3 supports macOS Apple Silicon Metal and Linux NVIDIA CUDA)"
fi

mkdir -p "$(dirname "$SRC_DIR")"
if [ -d "$SRC_DIR/.git" ]; then
    echo "==> Updating TFQ in ${SRC_DIR} (${REF})..."
    git -C "$SRC_DIR" fetch --depth 1 origin "$REF"
    git -C "$SRC_DIR" checkout -q FETCH_HEAD
else
    echo "==> Cloning TFQ into ${SRC_DIR}..."
    git clone --depth 1 --branch "$REF" "$REPO" "$SRC_DIR"
fi

echo "==> Building TFQ3 GPU converters and tools..."
make -C "$SRC_DIR" ${MAKE_ARGS[@]+"${MAKE_ARGS[@]}"}

echo "==> Installing tfq3 into ${PREFIX}..."
make -C "$SRC_DIR" install PREFIX="$PREFIX" ${MAKE_ARGS[@]+"${MAKE_ARGS[@]}"}

echo ""
echo "Installed: ${PREFIX}/bin/tfq3"
case ":${PATH}:" in
    *":${PREFIX}/bin:"*)
        ;;
    *)
        echo ""
        echo "Add ${PREFIX}/bin to your PATH to run 'tfq3' from anywhere:"
        echo "  export PATH=\"${PREFIX}/bin:\$PATH\""
        ;;
esac
echo ""
echo "Get started:"
echo "  tfq3 doctor"
echo "  tfq3 Qwen/Qwen3.8-27B"
