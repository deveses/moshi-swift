"""Convert a BF16 Moshi MLX checkpoint to a mixed-precision (.mp.safetensors) checkpoint.

Policy v1 (matches MoshiLib/MixedPrecision.swift):
  q8 group=64: text_emb, text_linear, audio_embs.*, depformer.*
  q4 group=32: every other Linear / Embedding (bulk transformer MLP + attention).

We do NOT special-case first/last LM transformer layers. mlx-swift's
update(modules:) raises mismatchedContainers when the update tree has a sparse
array of recursive modules (Module.swift:629), so any layer-level sensitivity rule
makes pass 2's complement filter unusable on the Swift side. Sticking to non-layer
paths keeps both pass trees dense.

Two passes of `mlx.nn.quantize` with mutually-exclusive predicates. Pass 2 uses
the default predicate (`hasattr(m, "to_quantized")`), which silently skips
QuantizedLinear / QuantizedEmbedding produced by pass 1.
"""

import argparse

import mlx.core as mx
import mlx.nn as nn

import moshi_mlx


def is_sensitive(path: str) -> bool:
    if path in ("text_emb", "text_linear"):
        return True
    if path.startswith("audio_embs."):
        return True
    if path.startswith("depformer."):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("original_weights", type=str, help="path to BF16 .safetensors")
    parser.add_argument("--out", type=str, required=True, help="output .mp.safetensors")
    parser.add_argument(
        "--config",
        type=str,
        default="moshi_2024_07",
        help="moshi_2024_07 (alias for v0_1, the 7B Moshi), v0_1, 1b, 1b-16rvq, helium-2b",
    )
    args = parser.parse_args()

    if args.config in ("moshi_2024_07", "v0_1"):
        lm_config = moshi_mlx.models.config_v0_1()
    elif args.config == "1b":
        lm_config = moshi_mlx.models.config1b_202412()
    elif args.config == "1b-16rvq":
        lm_config = moshi_mlx.models.config1b_202412_16rvq()
    elif args.config == "helium-2b":
        lm_config = moshi_mlx.models.config_helium_1_preview_2b()
    else:
        raise ValueError(f"unknown config '{args.config}'")

    print(f"model config: {args.config} ({lm_config.transformer.num_layers} layers)")

    model = moshi_mlx.models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    print(f"loading weights {args.original_weights}")
    model.load_weights(args.original_weights, strict=True)
    print("weights loaded")

    print("pass 1: q8 (group 64) on sensitive modules")
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda p, m: is_sensitive(p) and hasattr(m, "to_quantized"),
    )

    print("pass 2: q4 (group 32) on remaining Linear / Embedding")
    nn.quantize(model, group_size=32, bits=4)

    print(f"saving mixed-precision weights to {args.out}")
    model.save_weights(args.out)


if __name__ == "__main__":
    main()
