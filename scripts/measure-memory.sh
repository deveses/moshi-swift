#!/usr/bin/env bash
# Run each Moshi variant under MoshiCLI and write per-phase memory snapshots
# to measurements/<date>/<variant>.jsonl (plus stdout/stderr to .log).
#
# Usage: scripts/measure-memory.sh [variant-glob]
#   no args     run all variants
#   "q4 q8"     run only the variants matching those names

set -uo pipefail

cd "$(dirname "$0")/.."

DATE=${DATE:-$(date +%Y-%m-%d)}
OUT_DIR="measurements/$DATE"
mkdir -p "$OUT_DIR"

CLI=./build/Build/Products/Release/MoshiCLI
if [[ ! -x "$CLI" ]]; then
    echo "CLI not built at $CLI — run 'make build' first." >&2
    exit 1
fi

MIMI_Q4=hf://kyutai/moshiko-mlx-q4/tokenizer-e351c8d8-checkpoint125.safetensors
MIMI_Q8=hf://kyutai/moshiko-mlx-q8/tokenizer-e351c8d8-checkpoint125.safetensors
MIMI_BF16=hf://kyutai/moshiko-mlx-bf16/tokenizer-e351c8d8-checkpoint125.safetensors

run_variant() {
    local label=$1
    shift
    local memlog="$OUT_DIR/$label.jsonl"
    local log="$OUT_DIR/$label.log"
    if [[ "$#" -gt 0 && -n "${SELECT:-}" ]] && ! [[ " $SELECT " == *" $label "* ]]; then
        return
    fi
    rm -f "$memlog"
    echo "==> $label"
    local start=$(date +%s)
    if ! "$CLI" "$@" --memlog "$memlog" > "$log" 2>&1; then
        echo "    $label exited non-zero — keeping partial $memlog (see $log)"
    fi
    echo "    elapsed $(( $(date +%s) - start ))s"
}

SELECT="${*:-}"

run_variant "moshi-1b-q6" \
    run "hf://lmz/moshi-swift/moshi-37c6cfd6@200.q6.safetensors"

run_variant "moshi-large-q4" \
    run "hf://kyutai/moshiko-mlx-q4/model.q4.safetensors" \
    --config moshi7b --mimi-model "$MIMI_Q4"

run_variant "moshi-large-q8" \
    run "hf://kyutai/moshiko-mlx-q8/model.q8.safetensors" \
    --config moshi7b --mimi-model "$MIMI_Q8"

if [[ "${SKIP_BF16:-0}" != "1" ]]; then
    run_variant "moshi-large-bf16" \
        run "hf://kyutai/moshiko-mlx-bf16/model.safetensors" \
        --config moshi7b --mimi-model "$MIMI_BF16"
fi

# Context sweep on q8.
for ctx in 2048 1536 1024; do
    run_variant "moshi-large-q8-ctx$ctx" \
        run "hf://kyutai/moshiko-mlx-q8/model.q8.safetensors" \
        --config moshi7b --mimi-model "$MIMI_Q8" --main-context "$ctx"
done

echo "Done. Output under $OUT_DIR/"
