# Memory Footprint Brainstorm

Goal: make the larger Moshi checkpoints, especially q8 and BF16, usable on Macs with 8 GB RAM.

This is a brainstorming document, not a committed implementation plan. The ideas are grouped by likely engineering cost and risk.

## Current Situation

The app can select several Moshi variants:

- Moshi 1B q6 from `lmz/moshi-swift`
- Moshi q4 from `kyutai/moshika-mlx-q4`
- Moshi q8 from `kyutai/moshika-mlx-q8`
- Moshi BF16 from `kyutai/moshika-mlx-bf16`

The larger q8 and BF16 variants are difficult on 8 GB machines because memory is consumed by:

- model weights
- loaded safetensors before/while updating model parameters
- main transformer KV cache
- Mimi codec weights and streaming state
- depformer weights/cache
- temporary MLX arrays during warmup and generation
- app/UI/runtime overhead

The main win probably needs to combine several reductions rather than rely on one trick.

## 1. Avoid Loading Duplicate Weight Copies

Priority: high  
Risk: medium  
Expected impact: high during model load

The current loading path does:

1. `loadArrays(url:)`
2. `ModuleParameters.unflattened(weights)`
3. `model.update(parameters:)`
4. `eval(model)`

For large models this may temporarily hold both the safetensors arrays and the model parameters. On an 8 GB Mac, peak memory during load may fail before steady-state inference would.

Ideas:

- Investigate whether MLX Swift supports lazy or memory-mapped safetensors loading.
- Release intermediate dictionaries as soon as possible after `model.update`.
- Split loading into phases if the API allows partial updates.
- Avoid `let weights` / `let parameters` lifetime extension across later setup work.
- Add autorelease scopes around loading paths in app and CLI.

Possible target files:

- `Moshi/ContentView.swift`
- `MoshiCLI/RunMoshi.swift`
- `MoshiCLI/RunMimi.swift`
- `MoshiCLI/CLI.swift`

## 2. Load Moshi Before Mimi, Then Free Intermediates

Priority: high  
Risk: low  
Expected impact: medium to high during startup

Moshi and Mimi are both needed for full duplex generation, but their loading can be staged carefully.

Current model construction loads Moshi, then Mimi, then vocab, then warmups. This is mostly good, but we should ensure intermediate Moshi load arrays are released before Mimi load begins.

Ideas:

- Wrap `makeMoshi` internals in `do { ... }` scopes.
- Explicitly call `eval(model)` after update, then let weight dictionaries fall out of scope.
- Delay Mimi warmup until Moshi warmup has completed and temporary arrays are gone.
- Consider optional "text-only smoke test" mode for loading larger checkpoints without Mimi.

## 3. Reduce Main Transformer Context

Priority: high  
Risk: low to medium  
Expected impact: medium

The large Moshi config uses a main transformer context of 3000. KV cache size grows linearly with context.

Approximate fp16 KV cache sizes:

- Moshi 1B, context 3000: about 375 MiB
- Moshi 7B-style q4/q8/BF16 config, context 3000: about 1.5 GiB

This is runtime memory, not weight memory. Reducing context can make the difference on 8 GB machines.

Ideas:

- Add app setting for "context budget": 512, 1024, 1536, 2048, 3000.
- Default 8 GB machines to 1024 or 1536.
- Use shorter context only for large models.
- Keep depformer context unchanged; it is already tiny.

Implementation sketch:

- Add a helper that returns a copy of `LmConfig` with `cfg.transformer.context` changed.
- Add a preset-level context override for q8 and BF16.
- Surface the selected context in the UI model details.

Tradeoff:

- Less conversational memory and possibly worse long-range coherence.
- Likely acceptable for first "make it run" mode.

## 4. Use Sliding Cache Allocation Instead of Full Growth

Priority: medium  
Risk: medium  
Expected impact: medium

`KVCacheSimple` grows storage in chunks and returns the active prefix. For a fixed maximum usable context, a rotating cache can cap memory.

Ideas:

- Allow main LM to use `RotatingKVCache`.
- Or add a `SlidingKVCache` for non-Mimi transformers.
- Keep only the most recent N tokens once context budget is exceeded.

This is conceptually similar to the existing Mimi rotating cache, but the causal mask and offset semantics must be checked carefully for the main LM.

## 5. Quantize KV Cache

Priority: medium  
Risk: medium to high  
Expected impact: medium to high

Even if the user is focused on weights, the KV cache is a big part of memory for q8/BF16 large models. A 1.5 GiB fp16 KV cache is expensive on an 8 GB Mac.

Options:

- Start with 8-bit KV cache.
- Try 4-bit keys/values after quality tests.
- Keep recent tokens fp16 and older tokens quantized.
- Quantize only values first, then keys.
- Explore TurboQuant-style vector quantization for keys, where inner products matter most.

Practical first version:

- Store compressed K/V in cache.
- Dequantize to float before `scaledDotProductAttention`.
- This reduces persistent memory but may add transient memory and latency.

Better long-term version:

- Custom attention path that consumes quantized K/V directly.

## 6. Mixed Precision Weight Policy

Priority: high  
Risk: medium  
Expected impact: high for custom checkpoints

Instead of using a uniform q4/q8/BF16 model, use a mixed policy:

- Main transformer MLP and attention projections: q4 or q6
- Embeddings: q8 or BF16
- Output heads: q8 or BF16
- First and last transformer layers: q8 or BF16
- Depformer: q8 or q6
- Mimi: q8 or BF16, depending on audio quality

This may preserve more quality than all-q4 while using much less memory than all-q8/BF16.

Possible implementation paths:

- Runtime selective `quantize(model:filter:)` before loading compatible weights.
- Offline conversion script that produces mixed safetensors.
- Multiple app presets: "large balanced", "large low memory", "large quality".

Important note:

Runtime `quantize(model:)` must match the checkpoint tensor format. If the checkpoint is already q8 or q4, the module structure must match those tensors before `update(parameters:)`.

## 7. Keep Mimi Small or Optional

Priority: medium  
Risk: low to medium  
Expected impact: medium

Full Moshi needs Mimi, but Mimi also consumes memory. For testing model load or text behavior, we can avoid it.

Ideas:

- Add a "load Moshi only" diagnostic mode.
- Add a text-only mode for q8/BF16 memory experiments.
- Use the smallest compatible Mimi checkpoint for all large variants if quality allows.
- Avoid loading Mimi until microphone/audio generation is actually started.

Tradeoff:

- Not full duplex until Mimi is loaded.
- Useful for debugging and staged startup.

## 8. Reduce Warmup Peak Memory

Priority: medium  
Risk: low  
Expected impact: medium during startup

Warmup allocates representative arrays for Mimi and Moshi. On constrained machines, warmup can push memory over the edge.

Ideas:

- Make warmup optional for q8/BF16 on 8 GB machines.
- Warm up one component at a time and force evaluation/release between them.
- Use smaller warmup inputs.
- Add a "slow start / low memory" mode that skips aggressive warmup.

Tradeoff:

- First generated tokens/audio frames may be slower.
- Better than failing to load.

## 9. Explicit Memory Cleanup Between Model Switches

Priority: medium  
Risk: low  
Expected impact: medium

The app already sets `loadState = .idle` before loading a different model. We can make this more deliberate.

Ideas:

- Reset output buffers, traces, stats, and temporary URLs before loading.
- Ensure audio player/microphone resources are stopped before switching.
- Add a short delay after releasing a model before loading the next.
- Investigate MLX cache clearing APIs, if available.

Target:

- Avoid model A and model B overlapping in memory while switching presets.

## 10. Avoid Large Trace / Token Buffers in Low Memory Mode

Priority: low to medium  
Risk: low  
Expected impact: small to medium

The app records traces and token streams. These are useful, but not essential for running a large model on 8 GB.

Ideas:

- Add "low memory mode" that disables trace collection.
- Cap output token/audio token history.
- Avoid keeping long `output` strings for long sessions.

## 11. App Presets for Memory Classes

Priority: high  
Risk: low  
Expected impact: usability

Add clear presets:

- `Moshi q4`: default large model for 8 GB
- `Moshi q8 low memory`: reduced context, optional warmup, trace off
- `Moshi BF16 diagnostic`: text-only or reduced context warning
- `Moshi BF16 full`: only recommended for larger RAM machines

The app can detect physical memory and warn when a selected model is unlikely to fit.

Potential check:

- Use `ProcessInfo.processInfo.physicalMemory`.
- If <= 8 GB, default q8/BF16 to low memory settings.

## 12. Offline Checkpoint Conversion

Priority: medium  
Risk: medium  
Expected impact: high

For serious memory reduction, generate custom checkpoints offline instead of doing everything at app startup.

Ideas:

- Convert BF16 to q6/q8 mixed policy.
- Skip or quantize specific submodules differently.
- Store only the tensors needed by the selected app mode.
- Create "8GB Mac" checkpoint variants.

Benefits:

- Lower startup peak memory.
- Less runtime logic.
- Easier to test reproducibly.

Costs:

- Need conversion tooling.
- Need upload/distribution path.
- Need validation against audio/text quality.

## 13. Evaluate Memory With Instrumentation

Priority: high  
Risk: low  
Expected impact: decision quality

Before deeper changes, measure memory at each phase:

- before model download
- after safetensors load
- after `ModuleParameters.unflattened`
- after `model.update`
- after `eval(model)`
- after Mimi load
- after warmup
- after 100 generation steps

Add lightweight logging in debug builds:

- resident memory
- MLX active memory
- MLX cache memory
- peak memory

The app already shows GPU memory stats; we should also log phase boundaries during loading.

## Suggested Roadmap

### Phase 1: Make Startup Less Spiky

- Scope/release intermediate weight dictionaries.
- Disable or shrink warmup for q8/BF16 low-memory mode.
- Add memory logging around load phases.
- Add 8 GB warning in the app.

### Phase 2: Lower Runtime Memory

- Add context budget setting.
- Default q8/BF16 to 1024 or 1536 context on 8 GB machines.
- Disable trace retention in low-memory mode.

### Phase 3: Better Quality Per Byte

- Build mixed precision checkpoint policy.
- Keep sensitive layers q8/BF16 and quantize bulk layers q4/q6.
- Validate speech quality and latency.

### Phase 4: Experimental Cache Compression

- Add 8-bit KV cache.
- Try q4/TurboQuant-style K/V cache.
- Consider custom attention kernels only if dequantization overhead is too high.

## Most Promising Combination

For an 8 GB Mac, the practical first target should be:

- q4 for normal usage
- q8 low-memory mode with reduced context
- BF16 as diagnostic or larger-RAM mode
- staged loading with lower peak memory
- optional warmup
- no trace retention in low-memory mode

Trying to make full BF16 large Moshi run comfortably on 8 GB may be unrealistic without reducing context, skipping components, or moving to mixed/custom quantized weights.
