# Memory Footprint Tasks

Tasks derived from [MEMORY_FOOTPRINT_BRAINSTORM.md](../MEMORY_FOOTPRINT_BRAINSTORM.md). Phase 0 + Phase 1 (01–05) are delivered; Phase 2 (06–08) is drafted but not started.

The intent is one task = one PR. Each file states a goal, a verifiable success criterion, and a file-level plan. Estimated savings are not commitments — they exist to set priority order; Task 2 replaces them with measurements.

## Status

| # | Task | Status | Depends on | Brainstorm sections |
| --- | --- | --- | --- | --- |
| [01](01-memory-instrumentation.md) | Memory instrumentation at phase boundaries | **done** 2026-05-15 | — | §13 |
| [02](02-baseline-measurements.md) | Baseline measurements (q4, q8, 1B q6, q8 context sweep) | **done** 2026-05-15 (BF16 deferred) | 01 | §13 |
| [03](03-release-intermediate-weights.md) | Align CLI load order with the app | **done** 2026-05-15 | 02 | §1, §2 |
| [04](04-optional-warmup.md) | Optional / shrunken warmup for low-memory mode | **done** 2026-05-15 | 01 | §8 |
| [05](05-platform-aware-memory-warning.md) | Platform-aware memory warning in app | **done** 2026-05-15 (GUI test pending) | 02 | §11 |
| [06](06-rotating-kv-cache.md) | Enable RotatingKVCache on the main LM | **done** 2026-05-15 | 02 | §4 |
| [07](07-trace-token-retention.md) | Cap trace and token retention in low-memory mode | **done** 2026-05-15 | 01 | §10 |
| [08](08-mlx-cache-limit.md) | Expose `MLX.GPU.set(cacheLimit:)` as a config knob | not started | 01 | §9 |
| [09](09-rotating-kv-cache-wrap-fix.md) | Investigate "wrap-correctness bug" (closed: misdiagnosed) | **closed** 2026-05-15 | 06 | §4 |

Phase 2 expected gain (estimates; replace with measurements as tasks land):

- **06** caps KV cache at `context × 8 MiB` (for q8 large). Functional sliding window confirmed: ctx=1024 holds ~512 MiB vs `KVCacheSimple`'s ~1.5 GB at step 3000. **But:** Moshi is trained at `context = 3000`; reducing the visible window via either `--main-context` or the rotating cache degrades output quality regardless of cache. The memory win is real for sessions under the trained context, where `KVCacheSimple` already works fine.
- **07** saves 50–200 MB at long sessions by capping per-step accumulators.
- **08** is a tradeoff knob (memory ↔ latency); useful in combination with 06 and 07 on tight machines.

Phase 2 does **not** lower steady-state memory for q8 large on initial load (already 8.00 GB at step 100). For that, Phase 3 (mixed-precision checkpoints, brainstorm §6 + §12) is the real lever — out of scope for the current task batch.

Baseline report: [MEMORY_BASELINE_2026-05-15.md](../MEMORY_BASELINE_2026-05-15.md). Key findings (see report for details):

- `--main-context` has zero effect on KV memory in the current implementation; brainstorm §3 was wrong, the real lever is §4 (RotatingKVCache).
- `loadArrays` is mmap'd; resident stays <50 MB through the entire pre-`eval` pipeline. Task 03's dict-release work was dropped — only the CLI load-order swap remains.
- q8 large peaks at exactly 8.00 GB resident; q4 large at 4.88 GB; 1B q6 at 1.75 GB.
