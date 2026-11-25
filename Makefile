CXX      := clang++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter
OBJCXX   := clang++ -O3 -std=c++17 -fobjc-arc -framework Foundation -framework Metal
METAL    := xcrun -sdk macosx metal
METALLIB := xcrun -sdk macosx metallib
BUILD    := build

.PHONY: all qwen27 flashnext glm tools test clean

all: tools qwen27 flashnext glm

$(BUILD):
	@mkdir -p $(BUILD)

tools: $(BUILD)/tfq3-inspect

qwen27: $(BUILD)/tfq3-qwen27-convert $(BUILD)/tfq3-qwen27-convert.metallib $(BUILD)/tfq3-qwen27-package

flashnext: $(BUILD)/tfq3-flashnext-convert $(BUILD)/tfq3-flashnext-convert.metallib

glm: $(BUILD)/tfq3-glm-convert $(BUILD)/tfq3-glm-convert.metallib

$(BUILD)/tfq3-inspect: src/tools/tfq3_inspect.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/tfq3-qwen27-convert: src/converters/qwen27/convert.mm | $(BUILD)
	$(OBJCXX) $< -o $@

$(BUILD)/tfq3-qwen27-convert.metallib: src/converters/qwen27/convert.metal | $(BUILD)
	$(METAL) -c $< -o $(BUILD)/qwen27.air
	$(METALLIB) $(BUILD)/qwen27.air -o $@
	@rm -f $(BUILD)/qwen27.air

$(BUILD)/tfq3-qwen27-package: src/converters/qwen27/package.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

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

test: all
	@python3 reference/tfq3_reference.py --self-test
	@$(BUILD)/tfq3-inspect -h >/dev/null 2>&1 || true
	@echo "All TFQ3 converters built cleanly in $(BUILD)/"

clean:
	rm -rf $(BUILD)
