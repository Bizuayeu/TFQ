UNAME_S   := $(shell uname -s)
CXX       ?= clang++
ifeq ($(UNAME_S),Linux)
CXX       := g++
endif
CXXFLAGS  := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter
OBJCXX    := clang++ -O3 -std=c++17 -fobjc-arc -framework Foundation -framework Metal
METAL     := xcrun -sdk macosx metal
METALLIB  := xcrun -sdk macosx metallib
NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_ARCH ?= native
NVCCFLAGS := -O3 -std=c++17 -arch=$(CUDA_ARCH) --use_fast_math
BUILD     := build

.PHONY: all metal cuda qwen27 flashnext glm tools test test-cuda clean

ifeq ($(UNAME_S),Darwin)
all: tools metal
else
all: tools cuda
endif

$(BUILD):
	@mkdir -p $(BUILD)

tools: $(BUILD)/tfq3-inspect $(BUILD)/tfq3-qwen27-package

metal: qwen27 flashnext glm

cuda: tools $(BUILD)/tfq3-qwen27-convert-cuda $(BUILD)/tfq3-flashnext-convert-cuda $(BUILD)/tfq3-glm-convert-cuda $(BUILD)/tfq3-cuda-test

qwen27: $(BUILD)/tfq3-qwen27-convert $(BUILD)/tfq3-qwen27-convert.metallib $(BUILD)/tfq3-qwen27-package

flashnext: $(BUILD)/tfq3-flashnext-convert $(BUILD)/tfq3-flashnext-convert.metallib

glm: $(BUILD)/tfq3-glm-convert $(BUILD)/tfq3-glm-convert.metallib

$(BUILD)/tfq3-inspect: src/tools/tfq3_inspect.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/tfq3-qwen27-package: src/converters/qwen27/package.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

# --- Apple Silicon Metal Targets ---
$(BUILD)/tfq3-qwen27-convert: src/converters/qwen27/convert.mm | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-qwen27-convert.metallib: src/converters/qwen27/convert.metal | $(BUILD)
	$(METAL) -c $< -o $(BUILD)/qwen27.air
	$(METALLIB) $(BUILD)/qwen27.air -o $@
	@rm -f $(BUILD)/qwen27.air

$(BUILD)/tfq3-flashnext-convert: src/converters/flashnext/convert.mm | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-flashnext-convert.metallib: src/converters/flashnext/convert.metal | $(BUILD)
	$(METAL) -c $< -o $(BUILD)/flashnext.air
	$(METALLIB) $(BUILD)/flashnext.air -o $@
	@rm -f $(BUILD)/flashnext.air

$(BUILD)/tfq3-glm-convert: src/converters/glm/convert.mm | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-glm-convert.metallib: src/converters/glm/convert.metal | $(BUILD)
	$(METAL) -c $< -o $(BUILD)/glm.air
	$(METALLIB) $(BUILD)/glm.air -o $@
	@rm -f $(BUILD)/glm.air

# --- NVIDIA CUDA Targets (sm_80+: Ampere, Ada, Hopper, Blackwell / GB10) ---
$(BUILD)/tfq3-qwen27-convert-cuda: src/converters/qwen27/convert.cu src/converters/cuda_encode_common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/tfq3-flashnext-convert-cuda: src/converters/flashnext/convert.cu src/converters/cuda_encode_common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/tfq3-glm-convert-cuda: src/converters/glm/convert.cu src/converters/cuda_encode_common.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BUILD)/tfq3-cuda-test: src/tools/tfq3_cuda_bench.cu src/kernels/cuda/tfq3_cuda.cuh | $(BUILD)
	$(NVCC) $(NVCCFLAGS) $< -o $@

test: all
	@python3 reference/tfq3_reference.py --self-test
	@if [ "$(UNAME_S)" != "Darwin" ] && [ -x "$(BUILD)/tfq3-cuda-test" ]; then $(BUILD)/tfq3-cuda-test; fi
	@echo "All TFQ3 targets built and verified in $(BUILD)/"

test-cuda: cuda
	@$(BUILD)/tfq3-cuda-test --bench

clean:
	rm -rf $(BUILD)
