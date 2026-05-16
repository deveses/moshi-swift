import Testing

@testable import moshi_lib

struct MixedPrecisionTests {

    @Test func sensitivePathsMatch() {
        #expect(isMixedPrecisionSensitive(path: "text_emb"))
        #expect(isMixedPrecisionSensitive(path: "text_linear"))
        #expect(isMixedPrecisionSensitive(path: "audio_embs.0"))
        #expect(isMixedPrecisionSensitive(path: "audio_embs.15"))
        #expect(isMixedPrecisionSensitive(path: "depformer.slices.0.linear_in"))
        #expect(
            isMixedPrecisionSensitive(
                path: "depformer.slices.7.transformer.layers.5.self_attn.in_proj"))
    }

    @Test func bulkPathsAreNotSensitive() {
        // All main-LM transformer.layers.* paths are bulk (q4) regardless of layer index.
        #expect(!isMixedPrecisionSensitive(path: "transformer.layers.0.self_attn.in_proj"))
        #expect(!isMixedPrecisionSensitive(path: "transformer.layers.15.gating.linear_in"))
        #expect(!isMixedPrecisionSensitive(path: "transformer.layers.31.self_attn.out_proj"))
        #expect(!isMixedPrecisionSensitive(path: "out_norm"))
    }
}
