import MLXNN

// Policy v1: sensitive paths get q8 group=64; everything else (bulk LM transformer
// Linear/Embedding) gets q4 group=32. We deliberately don't single out the first /
// last transformer layer: doing so would make pass 2's filter produce a sparse
// `transformer.layers` array, which mlx-swift's update(modules:) can't apply
// (Module.swift:629 throws mismatchedContainers on .none entries inside an array
// of recursive modules).
public func isMixedPrecisionSensitive(path: String) -> Bool {
    if path == "text_emb" || path == "text_linear" { return true }
    if path.hasPrefix("audio_embs.") { return true }
    if path.hasPrefix("depformer.") { return true }
    return false
}

public func applyMixedPrecisionPolicy(model: LM) {
    // Pass 1: q8 group=64 on sensitive Linear/Embedding (none under transformer.layers).
    quantize(model: model, groupSize: 64, bits: 8) { path, _ in
        isMixedPrecisionSensitive(path: path)
    }
    // Pass 2: q4 group=32 on bulk Linear/Embedding.
    //
    // The filter must exclude sensitive paths even though those are now
    // QuantizedLinear/QuantizedEmbedding. Unlike Python MLX, mlx-swift's
    // QuantizedLinear inherits from Linear (which is Quantizable), so a
    // permissive filter would re-quantize pass-1 outputs into broken q4 modules.
    quantize(model: model, groupSize: 32, bits: 4) { path, _ in
        !isMixedPrecisionSensitive(path: path)
    }
}
