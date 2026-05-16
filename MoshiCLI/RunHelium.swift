// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import AVFoundation
import Foundation
import MLX
import MLXNN
import MoshiLib

func makeHelium(_ url: URL, _ cfg: LmConfig) throws -> LM {
    let weights = try loadArrays(url: url)
    MemoryLog.shared.snapshot("after-loadArrays-moshi")
    let parameters = ModuleParameters.unflattened(weights)
    MemoryLog.shared.snapshot("after-unflatten-moshi")
    let model = LM(cfg, bSize: 1)
    if url.lastPathComponent.hasSuffix("q4.safetensors") {
        quantize(model: model, groupSize: 64, bits: 4)
    } else if url.lastPathComponent.hasSuffix("q6.safetensors") {
        quantize(model: model, groupSize: 64, bits: 6)
    } else if url.lastPathComponent.hasSuffix("q8.safetensors") {
        quantize(model: model, groupSize: 64, bits: 8)
    }
    MemoryLog.shared.snapshot("after-quantize-moshi")
    try model.update(parameters: parameters, verify: [.all])
    MemoryLog.shared.snapshot("after-update-moshi", kvCacheBytes: model.kvCacheMemoryBytes())
    eval(model)
    MemoryLog.shared.snapshot("after-eval-moshi", kvCacheBytes: model.kvCacheMemoryBytes())
    return model
}

func runHelium(
    _ url: URL, cfg: LmConfig, warmup: WarmupMode = .full, lowMemory: Bool = false
) throws {
    let stats = PerfStats()
    stats.lowMemory = lowMemory
    let helium = try makeHelium(url, cfg)
    let vocab = try loadVocab(cfg)
    MemoryLog.shared.snapshot("after-loadVocab")
    helium.warmup(warmup)
    MemoryLog.shared.snapshot("after-warmup-moshi", kvCacheBytes: helium.kvCacheMemoryBytes())
    print("done warming up")

    let maxSteps = helium.cfg.transformer.maxSeqLen
    let sampler = Sampler()

    var lastToken = MLXArray([1])
    for stepIdx in 0...maxSteps {
        let (textToken, _) = helium.sample(
            textIds: lastToken.reshaped([1, 1]), audioIds: [], stepIdx: stepIdx,
            textSampler: sampler,
            audioSampler: sampler, cb: stats)
        let textTokenI: Int = textToken[0].item()
        if var v = vocab[textTokenI] {
            if v == "<0x0A>" {
                print()
            } else {
                v.replace("▁", with: " ")
                print(v, terminator: "")
                fflush(stdout)
            }
        }
        lastToken = textToken
        if stepIdx == 100 {
            MemoryLog.shared.snapshot("after-step-100", kvCacheBytes: helium.kvCacheMemoryBytes())
        }
    }
}
