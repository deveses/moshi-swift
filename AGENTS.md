# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project

Experimental Swift/MLX implementations of Kyutai's Moshi (full-duplex spoken dialogue), Mimi (streaming neural audio codec, 24 kHz → 12.5 Hz / 1.1 kbps), and Hibiki (streaming speech-to-speech translation), with a CLI for macOS and a proof-of-concept iOS app. Built on [mlx-swift](https://github.com/ml-explore/mlx-swift), `swift-transformers`, and `swift-argument-parser`. Model weights are auto-downloaded from Hugging Face on first run.

## Build & Run

The project is an Xcode project (`moshi.xcodeproj`) — there is no SwiftPM `Package.swift`. Two schemes:
- `moshi-cli` → `MoshiCLI` (macOS command-line tool)
- `Moshi` → iOS/macOS SwiftUI app

The Makefile drives the CLI:
```bash
make build          # xcodebuild -scheme moshi-cli -derivedDataPath ./build
make run-1b         # MoshiCLI run            (1B Moshi model, default config)
make run-asr        # MoshiCLI run-asr        (speech recognition)
make run-mimi       # MoshiCLI run-mimi       (codec only)
make run-helium     # MoshiCLI run-helium     (Helium 2B text LM)
make run-qwen       # MoshiCLI run-qwen       (Qwen2 demo)
make format         # swift-format format --in-place --recursive .
```

The compiled binary lives at `./build/Build/Products/Release/MoshiCLI`. Subcommands take options like `--input mic` (microphone) or `--input path/to.wav`, `--config moshi1b|moshi7b`, `--channel N`. The `run-asr` model argument accepts either a local path or `hf://<repo>/<filename>` (see `maybeDownloadFromHub` in `MoshiCLI/CLI.swift`).

The iOS app is built/run from Xcode using the `Moshi` scheme (no Make target).

Tests: there are three test bundles wired in the Xcode project — `MoshiTests`, `MoshiUITests`, `MoshiLibTests` — but they currently contain only stub `@Test` cases. Run via `xcodebuild test -scheme moshi-cli` (or the `Moshi` scheme for app tests) if you add real coverage.

## Workarounds (from README)

- Xcode sets `LD_RUNPATH_SEARCH_PATHS` to include the executable path so the `moshi-lib` framework is found at runtime.
- Running from the CLI over SSH may need `security unlock-keychain` first.
- `OTHER_SWIFT_FLAGS` includes `-no-verify-emitter-module-interface` to work around [swift#64669](https://github.com/swiftlang/swift/issues/64669).

## Architecture

Three Xcode targets live side by side:

- **`MoshiLib/`** — pure-Swift library wrapping mlx-swift. This is where all the model code lives.
- **`MoshiCLI/`** — `argument-parser`-driven CLI. Each subcommand (`Run`, `RunMimi`, `RunHelium`, `RunQwen`, `RunAsr`, `AudioToCodes`, `CodesToAudio`) wires up audio I/O + a `MoshiLib` model. Weight loading and Hugging Face download helpers (`downloadFromHub`, `maybeDownloadFromHub`, `makeTokenizer`) live in `CLI.swift`.
- **`Moshi/`** — SwiftUI app (`moshiApp`, `ContentView`, `ModelView`, `AudioRT`, `DeviceStat`). Uses the same `MoshiLib`.

### MoshiLib model graph

The library is organized by model component, not by model variant — variants are just different `LmConfig`/`MimiConfig` values composed from the same building blocks:

- `Transformer.swift` — generic causal transformer used everywhere; `TransformerConfig` controls dims, RoPE, gating, KV-cache rotation, etc.
- `KVCache.swift` — KV caches, including the rotating variant used by Mimi's transformer and the TurboQuant KV-cache implementation used by the app toggle.
- `Conv.swift`, `Seanet.swift` — streaming 1D convolutions and the SEANet encoder/decoder used by Mimi.
- `Mimi.swift` — streaming neural audio codec. `MimiConfig.mimi_2024_07(numCodebooks:)` is the canonical config; `encodeStep`/`decodeStep` operate on `StreamArray` for true streaming.
- `LM.swift` — top-level `LM` module plus `Depformer` (per-codebook hierarchical sampler) and `LMGen` (generation loop). One `LM` class serves Moshi, Helium, and ASR; the variant is determined by `LmConfig` (`moshi1b`, `moshi_2024_07`, `helium2b`, `asr300m`/`asr1b`/`asr2b`).
- `ASR.swift` — speech-recognition entry points (`runAsr`, `runAsrMic`).
- `Qwen2.swift` — standalone Qwen2 implementation used by `run-qwen`.
- `Streaming.swift` — `StreamArray`, the optional-`MLXArray` wrapper used to thread "no data this step" through the pipeline.
- `Quantization.swift` — `quantize(model:groupSize:bits:)` applied based on weight-file suffix (`.q4/.q6/.q8.safetensors`) before parameter loading.
- `Perf.swift`, `Utils.swift` — timing/perf-stats helpers and utilities.

### TurboQuant KV cache notes

The SwiftUI app can load Moshi/ASR with a TurboQuant KV cache toggle in `Moshi/ModelView.swift`. The flag is threaded through `Evaluator.load`, `makeMoshi`, `LM`, `Depformer`, and `Transformer.makeCache`.

The current Swift TurboQuant path is a conservative 4-bit MLX-native affine cache in `KVCache.swift`. It uses QR rotation, QR sign correction, and per-token normalization with norms baked into quantized scales/biases. The reference implementation lives outside this repo at `/Users/slawomirstrumecki/Work/gh/deveses/turboquant-mlx`; use it as the source of truth when changing the algorithm. Do not switch the app default back to 3-bit unless QJL or equivalent quality protection is implemented.

When resetting an `LM`, reset both the main transformer cache and the depformer cache. Warmup must leave caches empty before real generation starts.

The app's Device tab displays KV cache memory through `LM.kvCacheMemoryBytes()`, using the cache arrays' logical shape and dtype size as a fallback when MLX reports zero bytes for lazy arrays.

### SwiftUI app notes

The stats panel in `ModelView.swift` intentionally uses one segmented `Picker` plus a `switch` over the selected page. Avoid nesting a `TabView` inside the custom stats panel; on macOS it adds native tab chrome and makes the panel look broken.

On macOS, closing the last window should quit the app. This behavior is implemented with an `NSApplicationDelegate` in `Moshi/moshiApp.swift`.

### Audio plumbing (CLI)

`MoshiCLI/Audio.swift` and `AudioRT.swift` provide `MicrophoneCapture` and `AudioPlayer` (24 kHz, mono). The typical mic loop in `RunMoshi.swift`/`ASR.swift` is: `mic → mimi.encodeStep → LMGen.step → mimi.decodeStep → player`.

### Weight loading convention

Weights are MLX `.safetensors`. Files whose name ends in `.q4/.q6/.q8.safetensors` are quantized — `make*` helpers in the Run files call `quantize(...)` with the matching `groupSize`/`bits` *before* `model.update(parameters:)`. ASR config is auto-detected from the shape of the `out_norm.weight` tensor (`CLI.swift:RunAsr.run`). Vocab JSONs are pulled from `lmz/moshi-swift` on the Hub keyed by `textOutVocabSize` (`RunMoshi.swift:loadVocab`).

## Style

- Formatter: `swift-format` with `lineLength: 100`, 4-space indent (`.swift-format`). Always run `make format` before committing.
- Per `CONTRIBUTING.md`: this is a research repo — bug fixes welcome, new-feature PRs and refactors generally are not.
