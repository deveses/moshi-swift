## 2026-05-15
- added TurboQuant KV cache toggle in model settings
- added KV cache memory usage to the Device stats tab
- reset depformer KV cache together with the main transformer cache
- improved TurboQuant cache quality with rotation sign correction, normalization, and 4-bit default
- changed macOS behavior to quit the app when the last window is closed
- cleaned up the app stats panel by replacing the nested tab view with a single segmented control
- added memory instrumentation: `MemorySnapshot` / `MemoryLog` in MoshiLib; CLI `--memlog` writes JSON Lines snapshots at every load / warmup / step boundary; app debug toggle in the settings popover
- added platform-aware memory budget warning before model load (`MemoryBudget.swift`); preset metadata declares estimated steady-state bytes from measured baselines; macOS shows a `sysctl iogpu.wired_limit_mb` hint
- added `WarmupMode { full, minimal, none }` and `--warmup` CLI flag / app picker; iOS defaults to `.minimal`, macOS to `.full`
- aligned CLI load order with the app (Moshi before Mimi in all `run*` paths)
- added `--main-context N` and `--rotating-kv-cache` flags / app toggle; `RotatingKVCache` rewritten as a functional sliding window (append + drop oldest)
- fixed `RotatingKVCache` buffer dtype: `Transformer.makeCache` previously read the `inProj.weight` dtype which is `uint32` for quantized models — caused 2× over-allocation; now hardcoded to `bfloat16` (the compute dtype)
- added `--low-memory` CLI flag / app toggle: skips per-token accumulators in `PerfStats`, caps event log at 1000, skips end-of-run trace / wav / codes writes, caps the app's displayed text buffer
- added `--mlx-cache-limit <bytes>` CLI flag / app input that calls `MLX.GPU.set(cacheLimit:)` before generation
- documented analysis and numbers in [MEMORY_FOOTPRINT_BRAINSTORM.md](MEMORY_FOOTPRINT_BRAINSTORM.md) and [MEMORY_BASELINE_2026-05-15.md](MEMORY_BASELINE_2026-05-15.md); reproducible via [scripts/measure-memory.sh](scripts/measure-memory.sh) + [scripts/aggregate-memory.py](scripts/aggregate-memory.py)

## 2026-05-10
- added q8, q16 and bf16 model selection
