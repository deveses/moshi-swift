# Phase 3 — Mixed-Precision Weight Policy for Moshi 7B

## Context

[MEMORY_FOOTPRINT_BRAINSTORM.md](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/MEMORY_FOOTPRINT_BRAINSTORM.md) identifies **mixed-precision weights** as the single largest remaining lever for fitting Moshi 7B onto an 8 GB Mac: ~2.0–2.4 GB savings on weights, larger than every other unlanded change combined. Phase 0 already produced a measured baseline ([MEMORY_BASELINE_2026-05-15.md](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/MEMORY_BASELINE_2026-05-15.md)) showing q8 steady-state at ~8.00 GB, dominated by 7.64 GB of weights.

The brainstorm explicitly rules out a pure-runtime solution: mlx-swift's `quantize(model:filter:)` rewrites the in-memory module graph *before* `model.update(parameters:)`, but the on-disk safetensors stores no bits/group_size metadata — only the structural `.weight` (uint32-packed) / `.scales` / `.biases` triplets per quantized layer. So the disk layout and the Swift-side module shape must be produced by **the same policy applied in lockstep**: a Python converter offline, and a matching Swift filter at load time.

User decisions captured up-front (this session):

- **Policy:** q4 bulk (MLP + attention projections) + q8 sensitive (embeddings, output head, first/last LM layer, depformer).
- **Distribution:** local file only for v1 — no HF repo yet.
- **Scope of this plan:** Python converter + Swift runtime hook + CLI wiring + manual validation + GUI integration (file-picker, both platforms). No HF upload.
- **Execution order:** CLI flow first (steps 1–5); GUI integration last (step 6). The GUI piece may be deferred as a follow-up after CLI validation if needed.

Goal of the plan: a CLI flow where the user can produce `model.mp.safetensors` from a BF16 input, load it via `moshi-cli` *and* the Moshi GUI app, and observe weight memory drop from ~7.64 GB to ~5.0 GB.

## Mixed-Precision Policy

Hardcoded in both Python and Swift, identical predicates. **Policy v1 for `LM` (config `moshi_2024_07`, 32 transformer layers):**

| Path pattern                              | Bits | Group | Rationale                              |
| ----------------------------------------- | ---- | ----- | -------------------------------------- |
| `text_emb`                                | 8    | 64    | Vocab embedding, quality-sensitive     |
| `text_linear`                             | 8    | 64    | Output head, quality-sensitive         |
| `audio_embs.0` … `audio_embs.15`          | 8    | 64    | Audio codebook embeddings              |
| `depformer.*` (all Linear + Embedding)    | 8    | 64    | Small, runs per-step, quality-critical |
| **All `transformer.layers.*` Linear**     | 4    | 32    | Bulk: MLP + attention in main LM       |
| Norms, conv layers (Mimi), non-projective | —    | —     | Untouched                              |

**Why no first/last-layer rule?** mlx-swift's `update(modules:)` raises
`mismatchedContainers` when the update tree contains a sparse array of recursive
modules ([Module.swift:629](build/SourcePackages/checkouts/mlx-swift/Source/MLXNN/Module.swift#L629)). Any layer-level
predicate would make pass 2's complementary filter produce a sparse
`transformer.layers` array. Skipping the special case keeps both passes dense and
costs only ~100 MB (2/32 transformer layers at q4 instead of q8). The
quality-critical paths (embeddings, output head, depformer) are still q8.

Mimi is loaded from its own (already q8) checkpoint and is **not** part of this work — keep it as-is.

This stacks with Scenario B in the brainstorm (~5.5 GB weight target) without touching KV cache, warmup, or trace retention — those land separately.

## Components to Build

### 1. Python converter — `scripts/convert_mixed_precision.py`

New file in [scripts/](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/scripts/) alongside the existing `eval_mlx.py`. Mirrors the shape of upstream `moshi/scripts/quantize_mlx.py` but applies two `mlx.nn.quantize` passes with mutually-exclusive `class_predicate` filters.

Shape:

```python
import mlx.core as mx, mlx.nn as nn
from moshi_mlx.models import LmConfig, Lm

cfg = LmConfig.from_name("moshi_2024_07")          # match Swift's LmConfig.moshi_2024_07()
model = Lm(cfg)
model.load_weights(args.in_file, strict=True)      # bf16 input
model.set_dtype(mx.bfloat16)

def is_sensitive(path: str, mod) -> bool:
    if path in ("text_emb", "text_linear"): return True
    if path.startswith("audio_embs."): return True
    if path.startswith("transformer.layers.0.") or path.startswith("transformer.layers.31."):
        return True
    if path.startswith("depformer."): return True
    return False

# Pass 1: q8 on sensitive Linear/Embedding
nn.quantize(model, group_size=64, bits=8,
            class_predicate=lambda p, m: is_sensitive(p, m) and isinstance(m, (nn.Linear, nn.Embedding)))
# Pass 2: q4 on everything else that's still a plain Linear/Embedding
nn.quantize(model, group_size=32, bits=4,
            class_predicate=lambda p, m: isinstance(m, (nn.Linear, nn.Embedding)))

model.save_weights(args.out_file)   # writes .mp.safetensors
```

CLI: `python scripts/convert_mixed_precision.py --in model.bf16.safetensors --out model.mp.safetensors`.

**Reuses:** `mlx.nn.quantize` (Apple), `moshi_mlx.models.LmConfig` (Kyutai upstream).

**Risk to verify during impl:** Python's `mlx.nn.quantize` skips already-`QuantizedLinear` modules between passes — but if it doesn't, the second pass would try to re-quantize. Mitigation in the predicate: `isinstance(m, (nn.Linear, nn.Embedding))` excludes already-quantized modules (which are `nn.QuantizedLinear` / `nn.QuantizedEmbedding`).

### 2. Swift filter helper — `MoshiLib/MixedPrecision.swift` (new)

New file in [MoshiLib/](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/MoshiLib/). Exports one function:

```swift
public func applyMixedPrecisionPolicy(model: LM) {
    let sensitive: (String, Module) -> Bool = { path, mod in
        if path == "text_emb" || path == "text_linear" { return true }
        if path.hasPrefix("audio_embs.") { return true }
        if path.hasPrefix("transformer.layers.0.") { return true }
        if path.hasPrefix("transformer.layers.31.") { return true }
        if path.hasPrefix("depformer.") { return true }
        return false
    }
    // Pass 1: q8 sensitive
    quantize(model: model, groupSize: 64, bits: 8, filter: sensitive)
    // Pass 2: q4 remainder (default filter = all; already-Quantized are silently skipped)
    quantize(model: model, groupSize: 32, bits: 4)
}
```

**Why this works:** mlx-swift's `quantize(model:groupSize:bits:filter:)` (in `MLXNN/Quantized.swift:32`) calls `quantizeSingle` per matched module, which only converts plain `Linear`/`Embedding` instances and returns `nil` for `QuantizedLinear` — confirmed by tests at `mlx-swift/Tests/MLXTests/ModuleTests.swift:589-616`. So pass 2's default predicate is safe; it only touches what pass 1 left alone.

**Last-layer index:** hardcoded `31` initially. If we want to generalize later, derive from `model.cfg.transformer.numLayers - 1`. Keep it hardcoded for v1 — single config, single use site.

### 3. Hook into `makeMoshi` — edit [MoshiCLI/RunMoshi.swift:11-30](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/MoshiCLI/RunMoshi.swift#L11-L30)

Insert a new branch in the suffix dispatch, immediately after the `.q8.safetensors` case:

```swift
} else if url.lastPathComponent.hasSuffix(".mp.safetensors") {
    applyMixedPrecisionPolicy(model: model)
}
```

That is the **only edit to existing code**. Same pattern as the current q4/q6/q8 branches. `MemoryLog.shared.snapshot("after-quantize-moshi")` on the next line already captures the post-quantize footprint.

### 4. CLI invocation (no code change)

No CLI patch needed. `maybeDownloadFromHub` ([MoshiCLI/CLI.swift:41-53](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/MoshiCLI/CLI.swift#L41-L53)) already routes any non-`hf://` string through `URL(fileURLWithPath:)`. The `Run` subcommand's positional `model: String` arg ([MoshiCLI/CLI.swift:117-119](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/MoshiCLI/CLI.swift#L117-L119)) accepts a local filesystem path directly.

Source checkpoint is already on disk under [models/kyutai/](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/models/) (BF16 from `kyutai/moshika-mlx-bf16`). Converter reads from there; CLI loads the produced file from anywhere on disk.

Concrete invocation:

```
make build
python scripts/convert_mixed_precision.py \
    --in  models/kyutai/moshika-mlx-bf16/model.safetensors \
    --out models/kyutai/moshika-mlx-mp/model.mp.safetensors
.build/release/moshi-cli run \
    models/kyutai/moshika-mlx-mp/model.mp.safetensors \
    --config moshi7b --input mic --memlog /tmp/mp.jsonl
```

The `--memlog` flag (already in [MoshiCLI/CLI.swift:86-94](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/MoshiCLI/CLI.swift#L86-L94)) dumps per-phase memory to JSONL for comparison with the baseline doc.

### 5. Tests — `MoshiLibTests/MixedPrecisionTests.swift` (new)

The existing test files in [MoshiLibTests/](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/MoshiLibTests/) are placeholders. Add **one** structural test:

```swift
func testMixedPrecisionAppliesExpectedTypes() {
    let cfg = LmConfig.moshi_2024_07()
    let model = LM(cfg, bSize: 1)
    applyMixedPrecisionPolicy(model: model)
    // Sample assertions on the in-memory module graph after policy:
    XCTAssertTrue(model.textEmb is QuantizedEmbedding)       // sensitive
    XCTAssertTrue(model.textLinear is QuantizedLinear)       // sensitive
    // Bulk layer should be QuantizedLinear but at q4/group=32, not q8
    // (introspect the QuantizedLinear's bits/groupSize properties)
}
```

Keep it minimal — one test that proves the predicate hits the right modules. Audio-quality validation is manual (listen test).

### 6. GUI integration — Moshi app (last task, can be split as follow-up)

User decisions: file-picker UX (SwiftUI `.fileImporter`); show on both macOS and iOS, lean on the existing memory-warning UI for iOS users.

a. **Suffix branch in the GUI's `makeMoshi`** — edit [Moshi/ContentView.swift:284-290](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/Moshi/ContentView.swift#L284-L290). Mirror the CLI edit:

```swift
} else if url.lastPathComponent.hasSuffix(".mp.safetensors") {
    applyMixedPrecisionPolicy(model: model)
}
```

b. **New preset `case moshiMixed`** in `ModelSelect` ([Moshi/ContentView.swift:34-43](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/Moshi/ContentView.swift#L34-L43)):

- `name`: "Moshi q4/q8 mixed"
- `description`: "Custom mixed-precision build — bulk weights at q4, embeddings + first/last layer + depformer at q8. Targets ~5.5 GB."
- `estimatedSteadyStateBytes: 5_500_000_000` — drives the existing memory warning in [Moshi/ModelView.swift:181-194](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/Moshi/ModelView.swift#L181-L194), so iOS users automatically get warned without new code.
- Add to `availableModels` at [Moshi/ContentView.swift:149](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/Moshi/ContentView.swift#L149).

c. **Extend `MoshiModelPreset`** with one field:

```swift
let requiresUserPickedFile: Bool   // default false; true for moshiMixed
```

`modelRepo` / `modelFilename` for the new preset are left empty strings; the file-picker path supersedes them.

d. **Persist picked URL via security-scoped bookmark.** Add a small helper (in a new `Moshi/PickedModelBookmark.swift`, ~40 LOC) keyed by preset name (e.g., `UserDefaults.standard` key `"picked-url.moshiMixed"`). Use `URL.bookmarkData(options: .withSecurityScope, ...)` so the URL survives app relaunch and works on iOS sandboxed access. macOS uses `.withSecurityScope`; iOS uses the default `[]` and the picker's transient scope is fine for the in-session load (which is all we need — the model is loaded once at startup).

e. **URL resolution in `Moshi.init`** ([Moshi/ContentView.swift:818-832](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/Moshi/ContentView.swift#L818-L832)): prepend a new branch:

```swift
if preset.requiresUserPickedFile {
    guard let picked = PickedModelBookmark.resolve(presetName: preset.name) else {
        throw CustomError("needsFilePicker")  // caught by UI; shows .fileImporter
    }
    url = picked
} else if let localURL = preset.localResourceName.flatMap({
    Bundle.main.url(forResource: $0, withExtension: "safetensors")
}) {
    url = localURL
} else {
    url = try await ev.downloadFromHub(id: preset.modelRepo, filename: preset.modelFilename)
}
```

f. **File-picker UI.** In the model card view ([Moshi/ModelView.swift](/Users/slawomirstrumecki/Work/gh/deveses/moshi-swift/Moshi/ModelView.swift)), wrap the load button with a `.fileImporter(isPresented: $showPicker, allowedContentTypes: [.init(filenameExtension: "safetensors")!])` triggered on the `needsFilePicker` error. On selection: call `PickedModelBookmark.save(url:presetName:)`, then retry the load. ~30 LOC of SwiftUI.

g. **iOS handling.** No `#if os(...)` gating — preset is visible everywhere; the existing `availableMemoryBytes` warning in `ModelView.swift` already fires for the ~5.5 GB estimate on small devices. The `.fileImporter` works on both platforms.

**Skip / defer trigger:** if Step 3 (CLI validation) reveals a quality regression on the mixed-precision checkpoint, do NOT ship the GUI preset until the policy is tuned. Easy to enforce by simply not adding `moshiMixed` to `availableModels`.

## Files Touched (Summary)

Paths are relative to the moshi-swift repo root.

| File                                       | Action | Size            | Notes                                              |
| ------------------------------------------ | ------ | --------------- | -------------------------------------------------- |
| `scripts/convert_mixed_precision.py`       | new    | ~60 LOC         |                                                    |
| `MoshiLib/MixedPrecision.swift`            | new    | ~25 LOC         |                                                    |
| `MoshiCLI/RunMoshi.swift`                  | edit   | +3 lines        | suffix branch                                      |
| `MoshiLibTests/MixedPrecisionTests.swift`  | new    | ~30 LOC         |                                                    |
| `Moshi/ContentView.swift`                  | edit   | ~30 lines       | preset, suffix branch, URL resolution              |
| `Moshi/ModelView.swift`                    | edit   | ~30 lines       | `.fileImporter` + retry on `needsFilePicker`       |
| `Moshi/PickedModelBookmark.swift`          | new    | ~40 LOC         |                                                    |
| `moshi.xcodeproj`                          | edit   | add 3 file refs |                                                    |

Total CLI-only (steps 1–5): ~120 new LOC, 3 edited lines.
Total with GUI (step 6): ~190 new LOC, ~60 edited lines, 3 new Xcode file refs.

## Out of Scope

- **HF Hub upload.** v1 produces a local `model.mp.safetensors`; the user keeps it on disk.
- **Mimi mixed-precision.** Mimi loads from its own checkpoint and is small; touching it complicates the change without much gain.
- **Helium / Moshi 1B.** Policy is hardcoded for the 32-layer 7B config. Other configs need their own predicate variant — defer.
- **Per-layer bits config file.** v1 hardcodes the policy in two places (Python + Swift). If a second policy ever ships, extract to a shared JSON spec then.
- **Auto-discovery of the mixed-precision file** in the GUI (e.g., scanning the HF cache). The user picks the file once via `.fileImporter`; the bookmark persists across launches.

## Verification

End-to-end test on a machine with an existing BF16 checkpoint:

1. **Produce the checkpoint.** Run `python scripts/convert_mixed_precision.py --in <kyutai/moshika-mlx-bf16 cached path>/model.safetensors --out /tmp/model.mp.safetensors`. Expect output ~5.5 GB (vs 14 GB BF16, vs 7.4 GB q8).
2. **Build the CLI.** `make build`. Test should pass: `xcodebuild test -scheme MoshiLib`.
3. **Run the CLI** with the local checkpoint path. Observe `MemoryLog` output (already prints to stderr at each phase):
   - `after-quantize-moshi` → before weight load, MLX active should be small (just module shells).
   - `after-update-moshi` → MLX active should be ~5.0 GB (vs measured ~7.6 GB for uniform q8 in the baseline doc).
4. **Audio quality check.** Pipe a fixed prompt through `runMoshiMic`, compare output to a q4 and q8 baseline run with the same seed. Subjective: voice intelligibility, no obvious artifacts.
5. **Latency sanity.** Token/audio-frame timing should match or beat q8 (q4 bulk is faster on Apple GPU). Anything slower than q4 is a regression to investigate.

The phase-boundary `MemoryLog.shared.snapshot(...)` calls in `makeMoshi` already capture the numbers needed for step 3 — no new instrumentation.

## Open Questions to Resolve at Implementation Time

1. Does `mlx-swift`'s `quantize()` correctly handle a single model holding `QuantizedLinear` at `(bits=8, group=64)` AND `(bits=4, group=32)` in different submodules? Tests confirm structural mixed precision works; the bits/group_size combination needs a quick `print(model)` after the two passes to confirm.
2. Does upstream `moshi_mlx.models.Lm` expose `save_weights` writing exactly the safetensors layout mlx-swift expects? The brainstorm notes both sides use the same MLX format, but verify with a tiny round-trip test before running the full converter.
3. **(GUI)** iOS `.fileImporter` returns a URL with transient security-scoped access — verify that loading the ~5.5 GB file finishes before the scope expires, or hold `startAccessingSecurityScopedResource()` for the entire load. macOS bookmarks with `.withSecurityScope` survive relaunch; iOS may need the picker on every launch (acceptable for v1).
