# Memory Baseline — 2026-05-15

Phase-by-phase memory measurements for the cached Moshi variants on a single machine. Replaces the back-of-envelope numbers in [MEMORY_FOOTPRINT_BRAINSTORM.md §"Estimated Savings for the q8 Model"](MEMORY_FOOTPRINT_BRAINSTORM.md) where they diverge.

Raw JSON Lines per run are under [measurements/2026-05-15/](measurements/2026-05-15/); stdout logs are in the same directory.

## Test machine

- Mac17,6 — Apple M5 Max — **128 GiB unified memory**, macOS 26.5 (25F71)
- The machine itself has ample headroom; the numbers below describe how much *the model* consumes, which is what the brainstorm cares about — not whether it fits on this specific Mac. A model that peaks at 8.00 GB here will be in trouble on an 8 GB Mac.

## Workload

- Single shared input: `lmz/moshi-swift/bria-24khz.mp3` (about 13 s of audio, 163 chunks of 1920 PCM samples).
- Each run executes the full file end-to-end; `after-step-100` snapshot fires at the 100th internal generation step inside the chunk loop.
- CLI is built once at `Release`. All weight files were already cached locally (no downloads during the run).
- Repeated runs in sequence; no machine restart between runs. Filesystem cache may therefore be warm for later runs (this matters most for `after-loadArrays-*` phases).

## Per-phase resident memory

| Phase | moshi-1b-q6 | moshi-large-q4 | moshi-large-q8-ctx1024 | moshi-large-q8-ctx1536 | moshi-large-q8-ctx2048 | moshi-large-q8 |
| --- | --- | --- | --- | --- | --- | --- |
| before-download | 9.7 MB | 9.6 MB | 9.6 MB | 9.7 MB | 9.7 MB | 9.6 MB |
| after-loadArrays-mimi | 22.3 MB | 39.8 MB | 22.1 MB | 22.2 MB | 22.2 MB | 22.1 MB |
| after-update-mimi | 22.8 MB | 40.2 MB | 22.7 MB | 22.7 MB | 22.6 MB | 22.6 MB |
| after-loadArrays-moshi | 23.5 MB | 41.0 MB | 23.7 MB | 23.7 MB | 23.7 MB | 23.7 MB |
| after-unflatten-moshi | 23.8 MB | 41.3 MB | 24.1 MB | 24.1 MB | 24.1 MB | 23.9 MB |
| after-quantize-moshi | 27.9 MB | 46.5 MB | 29.2 MB | 29.1 MB | 29.2 MB | 29.1 MB |
| after-update-moshi | 27.9 MB | 46.5 MB | 29.2 MB | 29.1 MB | 29.2 MB | 29.1 MB |
| after-eval-moshi | 1.39 GB | 4.52 GB | 7.64 GB | 7.64 GB | 7.64 GB | 7.64 GB |
| after-loadVocab | 1.41 GB | 4.53 GB | 7.65 GB | 7.65 GB | 7.65 GB | 7.65 GB |
| after-warmup-mimi | 1.74 GB | 4.87 GB | 7.99 GB | 7.99 GB | 7.99 GB | 7.99 GB |
| after-warmup-moshi | 1.74 GB | 4.87 GB | 7.99 GB | 7.99 GB | 7.99 GB | 7.99 GB |
| after-step-100 | 1.75 GB | 4.88 GB | 8.00 GB | 8.00 GB | 8.00 GB | 8.00 GB |

## Peaks across the whole run

| Variant | Peak resident | Peak MLX active | Peak MLX cache | MLX peak | KV cache @ step 100 |
| --- | --- | --- | --- | --- | --- |
| moshi-1b-q6 | 1.75 GB | 1.81 GB | 227.6 MB | 1.89 GB | 38.0 MB |
| moshi-large-q4 | 4.88 GB | 5.02 GB | 313.7 MB | 5.10 GB | 134.0 MB |
| moshi-large-q8-ctx1024 | 8.00 GB | 8.15 GB | 313.8 MB | 8.23 GB | 134.0 MB |
| moshi-large-q8-ctx1536 | 8.00 GB | 8.15 GB | 313.7 MB | 8.23 GB | 134.0 MB |
| moshi-large-q8-ctx2048 | 8.00 GB | 8.15 GB | 313.7 MB | 8.23 GB | 134.0 MB |
| moshi-large-q8 | 8.00 GB | 8.15 GB | 313.7 MB | 8.23 GB | 134.0 MB |

BF16 not measured; deferred (see "Missing" below).

## Key findings

### 1. `--main-context` has no effect on KV cache memory

The four q8 rows are identical to within 1 MB. The reason is in [KVCache.swift:73-91](MoshiLib/KVCache.swift#L73-L91): `KVCacheSimple` allocates storage in 256-step chunks (`step = 256`) up to `maxSeqLen` (4096). The `context` field on `TransformerConfig` is only consulted inside `Attention.callAsFunction` at [Transformer.swift:220-226](MoshiLib/Transformer.swift#L220-L226), where it bounds the *attention read window* — not the cache allocation. Reducing `context` from 3000 to 1024 changes which K/V slices `scaledDotProductAttention` sees, but the underlying tensor is the same size.

For the 7B config at step 100:

- 1 chunk × 256 slots × 32 layers × 32 KV heads × 128 head dim × (K+V) × 2 bytes (BF16 compute dtype) = **128 MiB** (134.2 × 10⁶ B; matches the measured 134.0 MB).
- At step 256+1 → 2 chunks → 256 MiB. At step 3000 → 12 chunks → 1.5 GiB.

**Brainstorm §3 ("Reduce Main Transformer Context", "context 3000 → 1024 saves ~1.0 GB") is wrong for the current implementation.** The actual lever is to use `RotatingKVCache` (brainstorm §4), which caps allocation at `context` slots. The new `--main-context` flag is therefore not useful for memory by itself — keep it as a documented config knob, but Task-style memory savings require §4.

### 2. Memory before `eval(model)` is trivial — Task 03's primary motivation evaporates

After `loadArrays`, `unflatten`, `quantize`, and `update`, resident is still **<50 MB** across all variants. The 1.4–7.7 GB jump happens at `eval(model)`. Two conclusions:

- `loadArrays` is effectively zero-cost in resident memory — strongly implies the safetensors file is mmap'd, with pages faulted in only as MLX touches them (or as the GPU materialises tensors during `eval`).
- The transient `[String: MLXArray]` and `ModuleParameters` dictionaries before `update` contribute ~0 MB to peak. The brainstorm's §1 ("Avoid Loading Duplicate Weight Copies") and the "wrap in scopes" portion of Task 03 should be deprioritised — there is nothing measurable to recover.

What survives in Task 03: the CLI load-order swap (Moshi before Mimi, for consistency with the app). It's a small consistency fix, not a memory win.

### 3. q8 large is right at the 8 GB Mac edge; q4 large fits with headroom

- **q8 large**: 8.00 GB peak resident. On an 8 GB Mac this will work via compressed memory + swap but with very little headroom for the OS, the app shell, or audio I/O buffers. Plausible but fragile.
- **q4 large**: 4.88 GB peak resident. ~3 GB of headroom on an 8 GB Mac — comfortable.
- **1B q6**: 1.75 GB peak resident. Runs anywhere.

### 4. The KV cache is small relative to weights — even at full context

At maxSeqLen 4096, the worst-case KV cache for the 7B config is ~2.1 GB (16 chunks × 128 MiB). At step 100 it's 134 MB. So the brainstorm's §3 talking about "1.5 GB of KV memory" is a maxSeqLen figure, not a steady-state one — and only reached at end-of-conversation. For short interactions (<256 steps) the KV cache is one chunk, ~134 MB.

This shifts priority: the **dominant** memory consumer in q8 large is the weights at 7.64 GB. Anything that reduces *that* (mixed-precision checkpoints, §6/§12) is worth far more than KV-cache work for typical conversation lengths.

## Discrepancies vs. brainstorm estimates

| Brainstorm claim | Estimate | Measured | Within 25%? |
| --- | --- | --- | --- |
| q8 weights ~7.4 GB (after `eval(model)`) | 7.4 GB | 7.64 GB | yes |
| q4 weights ~4.4–5.3 GB | 4.4–5.3 GB | 4.52 GB | yes |
| Main KV cache @ context 3000 BF16 dtype | 1.57 GB | n/a at step 100 (would be 1.5 GB at step 3000) | partial: matches at high step count, near-zero at low step count |
| Mimi weights + streaming state | 300–500 MB | ~330 MB (q8 row delta between `after-eval-moshi` and `after-warmup-mimi`) | yes |
| Activations / transient MLX buffers | 0.5–1 GB | ~150 MB (MLX peak − active for q8 large) | **measured 5–7× lower** |
| Total q8 steady-state | ~10 GB | 8.00 GB peak resident | **measured ~20% lower** |
| §3 "Reduce context 3000 → 1024 saves ~1.0 GB" | 1.0 GB | **0 GB with current implementation** | **claim invalid** |

## What this means for Phase 1 task ordering

- **Task 03** (release intermediate weight dicts; CLI load-order swap): the dict-release work is unnecessary — no measurable savings. Keep only the CLI load-order swap as a consistency fix.
- **Task 04** (optional warmup): still potentially useful. Warmup adds ~340 MB of MLX cache memory across all variants; skipping it on memory-tight machines is real savings but smaller than the brainstorm implied.
- **Task 05** (memory warning): now has concrete thresholds — q8 large at 8.00 GB peak should warn on any Mac with ≤8 GB RAM; q4 large is safe down to 8 GB.

## What this means for the brainstorm's later phases

- §3 "Reduce Main Transformer Context" must be paired with §4 "RotatingKVCache" to actually save memory; the context flag alone does nothing for allocation.
- §5 KV-cache quantization (TurboQuant, already built) saves ~134 MB at step 100 / up to ~1.5 GB at step 3000. Useful but smaller than the brainstorm's framing.
- §6 mixed-precision checkpoints remain the biggest q8 lever — the 7.64 GB of weights is the dominant cost.

## Warmup sweep (q4 large)

After Task 04 (`--warmup full|minimal|none`), measured on the same machine and workload. JSONL in [measurements/2026-05-15-warmup/](measurements/2026-05-15-warmup/).

| Mode | Resident @ step 100 | MLX cache @ warmup-moshi | MLX cache peak | Δ vs full (peak cache) |
| --- | --- | --- | --- | --- |
| full | 4981 MB | 313.7 MB | 314 MB | — |
| minimal | 4981 MB | 227.2 MB | 227 MB | **−87 MB** |
| none | 4980 MB | 0 MB | 98 MB | **−216 MB** |

Findings:

- **All three modes converge to the same resident memory at step 100** (~4981 MB) — `none` saves ~344 MB of MLX cache *at warmup time only*; that cache gets re-built during the first 100 real generation steps.
- **MLX cache peak is the actual lever.** Going `full → none` saves ~216 MB of peak cache; `full → minimal` saves ~87 MB.
- **Output quality is unchanged.** All three modes produce sensible English dialogue on the bria sample.
- **First-token latency**: total run time was 15 s for all three modes, dominated by 163 audio chunks of generation; the warmup difference is within timing noise at this granularity. A finer-grained latency probe would be needed to distinguish them (out of scope for the baseline).

Conclusion: Task 04's value on tight machines is ~216 MB of peak MLX cache (with `--warmup none`) — smaller than the brainstorm's §8 estimate of "0.2–0.5 GB peak" but in the same range. Useful on iOS where the per-process budget is tighter; less impactful on macOS.

## Rotating KV cache sweep (q8 large, Task 06)

13× repeated bria sample → ~2119 generation steps. Snapshots at steps 100, 256, 512, 1024, 2048. JSONL in [measurements/2026-05-15-rotating-fixed/](measurements/2026-05-15-rotating-fixed/) (post dtype-fix).

| Config | Wrap? | KV @ step 1024 | KV @ step 2048 | Final KV | Output coherence |
| --- | --- | --- | --- | --- | --- |
| Simple, ctx 3000 (baseline) | n/a | 512 MiB (4 chunks) | 1024 MiB (8 chunks) | 1542 MiB | coherent (repetitive) |
| Rotating, ctx 3000 | no (steps < 3000) | 1506 MiB | 1506 MiB | 1506 MiB | coherent |
| Rotating, ctx 2048 | yes after step 2048 | 1030 MiB | 1030 MiB | 1030 MiB | **degrades after wrap** |
| Rotating, ctx 1024 | yes after step 1024 | 518 MiB | 518 MiB | 518 MiB | **garbled soon after wrap** |

Two issues surfaced during this sweep:

### Dtype over-allocation (fixed)

The initial run of rotating ctx=1024 reported ~1036 MB instead of the expected ~512 MiB. Root cause: [Transformer.swift:325-328](MoshiLib/Transformer.swift#L325-L328) was reading the dtype from `selfAttn.inProj.weight.dtype`, which for a quantized model is the *storage* dtype (`uint32`, 4 bytes/element) rather than the *compute* dtype (`bf16`, 2 bytes/element). Fix: hardcode `bfloat16` for the RotatingKVCache allocation. Re-measured numbers above are post-fix.

### Wrap-correctness bug (unfixed)

`RotatingKVCache` produces garbled output once the buffer wraps (i.e., once `offset > maxSize`). The brainstorm's caveat — "the causal mask and offset semantics must be checked carefully for the main LM" — is real. The no-wrap case (ctx=3000 with ~2119 steps) produces normal English dialogue, identical in character to the simple-cache baseline. The wrap case (ctx=1024 wraps at step ~1024) starts degrading shortly after the wrap point.

Likely suspects (not yet diagnosed):

- In-place slice assignment `self.keys[..., currentOffset..<tMax] = ...` in `RotatingKVCache.update` may not be propagating the write through MLX's lazy evaluation graph the way the index-read assumes.
- RoPE position-offset interaction with the rotated buffer layout (each cell now carries the rotation at its *original* position, not its current buffer index — algorithmically this is correct but subtle, easy to get wrong by one).
- The always-on attention mask for `t == 1` in the rotating path (KVCacheSimple short-circuits this case).

A follow-up task ([09](tasks/09-rotating-kv-cache-wrap-fix.md)) tracks the correctness fix. Until it lands, rotating cache should not be advertised as a memory-reduction lever for long sessions — only as a no-wrap pre-allocation alternative for sessions known to stay under `context` steps.

### What ships from Task 06

- `--main-context N` and `--rotating-kv-cache` CLI flags on `Run`; `--repeat-input N` for measurement; multi-step snapshot milestones in the CLI.
- App toggle in [Moshi/ModelView.swift](Moshi/ModelView.swift) (mutually exclusive with TurboQuant).
- Dtype fix in `Transformer.makeCache` — gives a real 2× cut for the rotating allocation on quantized models.
- The wrap-correctness bug as a documented follow-up.

## Missing

- **BF16 large** — deferred. The brainstorm anticipates "may not fit on the test machine at all"; on this 128 GB machine it would fit easily, but a "fits-on-128GB" data point isn't useful for the 8 GB scenario. To capture, run `bash scripts/measure-memory.sh` without `SKIP_BF16=1`. Adding the row is a few minutes of wall time; deferred to keep this report focused on what's currently relevant.
- **Cold-start / warm-cache asymmetry** — the moshi-large-q4 row shows `after-loadArrays-mimi` at 39.8 MB vs ~22 MB elsewhere. This is the only large-variant run that didn't have the symlinked blob already in the filesystem cache. Future baselines should clear the OS page cache between runs (`sudo purge`) or record run order.
- **End-of-conversation KV cache** — only the step-100 point is captured. The brainstorm's "1.5 GB KV cache" is a step-3000 figure; would be useful to add a `after-step-3000` snapshot for a long-run scenario.
- **iOS measurements** — not run. iOS jetsam limits would change the threshold for Task 05.

## How to reproduce

```bash
make build
bash scripts/measure-memory.sh             # all variants including BF16
SKIP_BF16=1 bash scripts/measure-memory.sh # skip BF16
python3 scripts/aggregate-memory.py measurements/$(date +%Y-%m-%d)/
```
