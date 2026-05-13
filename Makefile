CONFIGURATION ?= Release
DERIVED_DATA ?= ./build
CLI := $(DERIVED_DATA)/Build/Products/$(CONFIGURATION)/MoshiCLI
MOSHI_1B_MODEL ?= hf://lmz/moshi-swift/moshi-37c6cfd6@200.q6.safetensors
MOSHI_Q4_MODEL ?= hf://kyutai/moshika-mlx-q4/model.q4.safetensors
MOSHI_Q8_MODEL ?= hf://kyutai/moshika-mlx-q8/model.q8.safetensors
MOSHIKA_Q4_MIMI_MODEL ?= hf://kyutai/moshika-mlx-q4/tokenizer-e351c8d8-checkpoint125.safetensors
MOSHIKA_Q8_MIMI_MODEL ?= hf://kyutai/moshika-mlx-q8/tokenizer-e351c8d8-checkpoint125.safetensors
ASR_MODEL ?= hf://lmz/moshi-swift/moshi-70f8f0ea@500.q8.safetensors

.PHONY: format run-1b run-1b-mic run-q4 run-q4-mic run-q8 run-q8-mic run-asr run-mimi run-helium run-qwen build

format:
	swift-format format --in-place --recursive .

run-1b: build
	$(CLI) run $(MOSHI_1B_MODEL)

run-1b-mic: build
	$(CLI) run $(MOSHI_1B_MODEL) --input mic

run-q4: build
	$(CLI) run $(MOSHI_Q4_MODEL) --config moshi7b --mimi-model $(MOSHIKA_Q4_MIMI_MODEL)

run-q4-mic: build
	$(CLI) run $(MOSHI_Q4_MODEL) --config moshi7b --mimi-model $(MOSHIKA_Q4_MIMI_MODEL) --input mic

run-q8: build
	$(CLI) run $(MOSHI_Q8_MODEL) --config moshi7b --mimi-model $(MOSHIKA_Q8_MIMI_MODEL)

run-q8-mic: build
	$(CLI) run $(MOSHI_Q8_MODEL) --config moshi7b --mimi-model $(MOSHIKA_Q8_MIMI_MODEL) --input mic

run-asr: build
	$(CLI) run-asr $(ASR_MODEL)

run-mimi: build
	$(CLI) run-mimi

run-helium: build
	$(CLI) run-helium

run-qwen: build
	$(CLI) run-qwen

build:
	xcodebuild -scheme moshi-cli -configuration $(CONFIGURATION) -derivedDataPath $(DERIVED_DATA)
