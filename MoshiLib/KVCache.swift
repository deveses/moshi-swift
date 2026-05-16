// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Parts of this file came from:
// https://github.com/ml-explore/mlx-swift-examples/blob/main/Libraries/LLM/KVCache.swift
// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLinalg
import MLXRandom

/// Interface for Key/Value cache for LLMs.
///
/// See ``LLMModel/newCache(parameters:)-47tyu``
public protocol KVCache: Evaluatable {

    /// get the current offset
    var offset: Int { get }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)
    func reset()
    func createAttentionMask(h: MLXArray) -> MLXArray?
}

func createAdditiveCausalMask(n: Int, offset: Int) -> MLXArray {
    let rinds = MLXArray(Int32(0)..<Int32(offset + n))
    let linds = offset != 0 ? MLXArray(Int32(offset)..<Int32(offset + n)) : rinds
    let mask = linds[0..., .newAxis] .< rinds[.newAxis]
    return mask * Float32(-1e9)
}

/// See https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/base.py#L11
class KVCacheSimple: KVCache, Evaluatable {
    let kHeadDim: Int
    let vHeadDim: Int
    let kvHeads: Int

    var keys: MLXArray?
    var values: MLXArray?

    var offset = 0
    var step = 256

    init(headDim: IntOrPair, kvHeads: Int) {
        self.kHeadDim = headDim.first
        self.vHeadDim = headDim.second
        self.kvHeads = kvHeads
    }

    public func reset() {
        self.keys = nil
        self.values = nil
        self.offset = 0
        self.step = 256
    }

    public func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous..<self.offset, 0...] = keys
        self.values?[.ellipsis, previous..<self.offset, 0...] = values

        return (
            self.keys![.ellipsis, ..<self.offset, 0...],
            self.values![.ellipsis, ..<self.offset, 0...]
        )
    }

    /// create an attention mask using the parameters from the KVCache.
    ///
    /// See also ``MultiHeadAttention/createAdditiveCausalMask(_:dtype:)`` -- same idea
    /// but doesn't honor the cache offset.
    func createAttentionMask(h: MLXArray) -> MLXArray? {
        let t = h.dim(1)
        if t > 1 {
            let rinds = MLXArray(Int32(0)..<Int32(offset + t))
            let linds = offset != 0 ? MLXArray(Int32(offset)..<Int32(offset + t)) : rinds
            let mask = linds[0..., .newAxis] .< rinds[.newAxis]
            return (mask * Float32(-1e9)).asType(h.dtype)
        }
        return nil
    }
}

class TurboQuantKVCache: KVCache, Evaluatable {
    let kHeadDim: Int
    let vHeadDim: Int
    let kvHeads: Int
    let bits: Int
    let groupSize: Int
    let useNormalization: Bool
    let rotationMatrix: MLXArray?

    var keyData: MLXArray?
    var keyScales: MLXArray?
    var keyBiases: MLXArray?
    var valueData: MLXArray?
    var valueScales: MLXArray?
    var valueBiases: MLXArray?

    var offset = 0
    var step = 256

    init(
        headDim: IntOrPair, kvHeads: Int, bits: Int = 3, groupSize: Int = 64,
        useRotation: Bool = true, useNormalization: Bool = true, seed: UInt64 = 42
    ) {
        self.kHeadDim = headDim.first
        self.vHeadDim = headDim.second
        self.kvHeads = kvHeads
        self.bits = bits
        self.groupSize = groupSize
        self.useNormalization = useNormalization
        self.rotationMatrix =
            useRotation ? Self.makeRotationMatrix(dim: headDim.first, seed: seed) : nil
    }

    public func reset() {
        self.keyData = nil
        self.keyScales = nil
        self.keyBiases = nil
        self.valueData = nil
        self.valueScales = nil
        self.valueBiases = nil
        self.offset = 0
        self.step = 256
    }

    public func innerState() -> [MLXArray] {
        [keyData, keyScales, keyBiases, valueData, valueScales, valueBiases].compactMap { $0 }
    }

    private static func makeRotationMatrix(dim: Int, seed: UInt64) -> MLXArray {
        let key = MLXRandom.key(seed)
        let random = MLXRandom.normal([dim, dim], dtype: .float32, key: key, stream: .cpu)
        let (q, r) = MLXLinalg.qr(random, stream: .cpu)
        let signs = sign(r.diag(stream: .cpu), stream: .cpu)
        let qCorrected = q * signs[.newAxis, 0...]
        eval(qCorrected)
        return qCorrected
    }

    private func normalizeForCache(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let norms = ((x * x).sum(axis: -1, keepDims: true) + 1e-8).sqrt()
        return (x / norms, norms)
    }

    private func quantizeForCache(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (B, H, T, _) = x.shape4
        let flat = x.flattened(end: -2)
        let (data, scales, biases) = MLX.quantized(flat, groupSize: groupSize, bits: bits)
        return (
            data.reshaped(B, H, T, -1),
            scales.reshaped(B, H, T, -1),
            biases.reshaped(B, H, T, -1)
        )
    }

    private func ensureCapacity(
        batchSize: Int, steps incomingSteps: Int, keyPackedDim: Int, keyScaleDim: Int,
        valuePackedDim: Int, valueScaleDim: Int, dataType: DType, scaleType: DType
    ) {
        let previous = self.offset
        if let keyData, previous + incomingSteps <= keyData.dim(2) {
            return
        }

        let nSteps = (step + incomingSteps - 1) / step
        let extraSteps = nSteps * step

        func expanded(_ current: MLXArray?, lastDim: Int, dtype: DType) -> MLXArray {
            let zeros = MLXArray.zeros([batchSize, kvHeads, extraSteps, lastDim], dtype: dtype)
            guard var current else {
                return zeros
            }
            if previous % step != 0 {
                current = current[.ellipsis, ..<previous, 0...]
            }
            return concatenated([current, zeros], axis: 2)
        }

        self.keyData = expanded(self.keyData, lastDim: keyPackedDim, dtype: dataType)
        self.keyScales = expanded(self.keyScales, lastDim: keyScaleDim, dtype: scaleType)
        self.keyBiases = expanded(self.keyBiases, lastDim: keyScaleDim, dtype: scaleType)
        self.valueData = expanded(self.valueData, lastDim: valuePackedDim, dtype: dataType)
        self.valueScales = expanded(self.valueScales, lastDim: valueScaleDim, dtype: scaleType)
        self.valueBiases = expanded(self.valueBiases, lastDim: valueScaleDim, dtype: scaleType)
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let (qKeys, qValues) = updateQuantized(keys: keys, values: values)
        var keys = dequantized(
            qKeys.data, scales: qKeys.scales, biases: qKeys.biases, groupSize: groupSize,
            bits: bits)
        var values = dequantized(
            qValues.data, scales: qValues.scales, biases: qValues.biases, groupSize: groupSize,
            bits: bits)
        if let rotationMatrix {
            keys = keys.matmul(rotationMatrix)
            values = values.matmul(rotationMatrix)
        }
        return (keys, values)
    }

    private func updateQuantized(keys: MLXArray, values: MLXArray)
        -> (
            keys: (data: MLXArray, scales: MLXArray, biases: MLXArray),
            values: (data: MLXArray, scales: MLXArray, biases: MLXArray)
        )
    {
        let previous = self.offset
        let incomingSteps = keys.dim(2)
        let keysForQuantization: MLXArray
        let keyNorms: MLXArray?
        let valuesForQuantization: MLXArray
        let valueNorms: MLXArray?
        if useNormalization {
            (keysForQuantization, keyNorms) = normalizeForCache(keys)
            (valuesForQuantization, valueNorms) = normalizeForCache(values)
        } else {
            keysForQuantization = keys
            keyNorms = nil
            valuesForQuantization = values
            valueNorms = nil
        }
        let keysToQuantize = rotationMatrix.map { keysForQuantization.matmul($0.T) }
            ?? keysForQuantization
        let valuesToQuantize = rotationMatrix.map { valuesForQuantization.matmul($0.T) }
            ?? valuesForQuantization

        var qKeys = quantizeForCache(keysToQuantize)
        var qValues = quantizeForCache(valuesToQuantize)
        if let keyNorms, let valueNorms {
            qKeys.1 = qKeys.1 * keyNorms
            qKeys.2 = qKeys.2 * keyNorms
            qValues.1 = qValues.1 * valueNorms
            qValues.2 = qValues.2 * valueNorms
        }
        ensureCapacity(
            batchSize: keys.dim(0),
            steps: incomingSteps,
            keyPackedDim: qKeys.0.dim(-1),
            keyScaleDim: qKeys.1.dim(-1),
            valuePackedDim: qValues.0.dim(-1),
            valueScaleDim: qValues.1.dim(-1),
            dataType: qKeys.0.dtype,
            scaleType: qKeys.1.dtype
        )

        self.offset += incomingSteps

        self.keyData?[.ellipsis, previous..<self.offset, 0...] = qKeys.0
        self.keyScales?[.ellipsis, previous..<self.offset, 0...] = qKeys.1
        self.keyBiases?[.ellipsis, previous..<self.offset, 0...] = qKeys.2
        self.valueData?[.ellipsis, previous..<self.offset, 0...] = qValues.0
        self.valueScales?[.ellipsis, previous..<self.offset, 0...] = qValues.1
        self.valueBiases?[.ellipsis, previous..<self.offset, 0...] = qValues.2

        return currentQuantized()
    }

    private func currentQuantized()
        -> (
            keys: (data: MLXArray, scales: MLXArray, biases: MLXArray),
            values: (data: MLXArray, scales: MLXArray, biases: MLXArray)
        )
    {
        (
            (
                self.keyData![.ellipsis, ..<self.offset, 0...],
                self.keyScales![.ellipsis, ..<self.offset, 0...],
                self.keyBiases![.ellipsis, ..<self.offset, 0...]
            ),
            (
                self.valueData![.ellipsis, ..<self.offset, 0...],
                self.valueScales![.ellipsis, ..<self.offset, 0...],
                self.valueBiases![.ellipsis, ..<self.offset, 0...]
            )
        )
    }

    func attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?,
        context: Int
    ) -> MLXArray {
        var (qKeys, qValues) = updateQuantized(keys: keys, values: values)
        let (B, queryHeads, querySteps, headDim) = queries.shape4
        let keySteps = qKeys.data.dim(2)
        let targetSteps = querySteps + min(context, keySteps - querySteps)
        var mask = mask

        if targetSteps < keySteps {
            let start = keySteps - targetSteps
            qKeys = (
                qKeys.data[.ellipsis, start..., 0...],
                qKeys.scales[.ellipsis, start..., 0...],
                qKeys.biases[.ellipsis, start..., 0...]
            )
            qValues = (
                qValues.data[.ellipsis, start..., 0...],
                qValues.scales[.ellipsis, start..., 0...],
                qValues.biases[.ellipsis, start..., 0...]
            )
        }

        if let m = mask {
            let maskLen = m.dim(-1)
            if qKeys.data.dim(2) < maskLen {
                let start = maskLen - qKeys.data.dim(2)
                mask = m[0..., start...]
            }
        }

        let repeats = queryHeads / kvHeads
        var rotatedQueries = queries * scale
        if let rotationMatrix {
            rotatedQueries = rotatedQueries.matmul(rotationMatrix.T)
        }
        if repeats > 1 {
            rotatedQueries = rotatedQueries.reshaped(B, kvHeads, repeats, querySteps, headDim)
        } else {
            rotatedQueries = rotatedQueries[0..., 0..., .newAxis, 0..., 0...]
        }

        let keyData = qKeys.data[0..., 0..., .newAxis, 0..., 0...]
        let keyScales = qKeys.scales[0..., 0..., .newAxis, 0..., 0...]
        let keyBiases = qKeys.biases[0..., 0..., .newAxis, 0..., 0...]
        var scores = quantizedMatmul(
            rotatedQueries, keyData, scales: keyScales, biases: keyBiases, transpose: true,
            groupSize: groupSize, bits: bits)

        if let mask {
            scores = scores + mask
        }

        let weights = softmax(scores, axis: -1, precise: true)
        let valueData = qValues.data[0..., 0..., .newAxis, 0..., 0...]
        let valueScales = qValues.scales[0..., 0..., .newAxis, 0..., 0...]
        let valueBiases = qValues.biases[0..., 0..., .newAxis, 0..., 0...]
        var output = quantizedMatmul(
            weights, valueData, scales: valueScales, biases: valueBiases, transpose: false,
            groupSize: groupSize, bits: bits)
        if let rotationMatrix {
            output = output.matmul(rotationMatrix)
        }
        return output.reshaped(B, queryHeads, querySteps, headDim)
    }

    func createAttentionMask(h: MLXArray) -> MLXArray? {
        let t = h.dim(1)
        if t > 1 {
            let rinds = MLXArray(Int32(0)..<Int32(offset + t))
            let linds = offset != 0 ? MLXArray(Int32(offset)..<Int32(offset + t)) : rinds
            let mask = linds[0..., .newAxis] .< rinds[.newAxis]
            return (mask * Float32(-1e9)).asType(h.dtype)
        }
        return nil
    }
}

// Sliding-window KV cache. Keeps only the most recent `maxSize` tokens by appending new K/V and
// dropping the oldest once the window is full. Functional (no in-place buffer mutation), so the
// MLX dependency graph stays simple. Memory at steady state is `maxSize` tokens; peak transient
// during update is `maxSize + t`.
class RotatingKVCache: KVCache, Evaluatable {
    var keys: MLXArray?
    var values: MLXArray?
    let maxSize: Int
    var offset: Int = 0

    init(bSize: Int, numHeads: Int, maxSize: Int, headDim: Int, dtype: DType) {
        _ = (bSize, numHeads, headDim, dtype)
        self.maxSize = maxSize
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let t = keys.dim(2)
        if let existingKeys = self.keys, let existingValues = self.values {
            var merged = concatenated([existingKeys, keys], axis: 2)
            var mergedV = concatenated([existingValues, values], axis: 2)
            if merged.dim(2) > self.maxSize {
                let drop = merged.dim(2) - self.maxSize
                merged = merged[.ellipsis, drop..., 0...]
                mergedV = mergedV[.ellipsis, drop..., 0...]
            }
            self.keys = merged
            self.values = mergedV
        } else {
            self.keys = keys
            self.values = values
        }
        self.offset += t
        return (self.keys!, self.values!)
    }

    func reset() {
        self.keys = nil
        self.values = nil
        offset = 0
    }

    func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    func createAttentionMask(h: MLXArray) -> MLXArray? {
        let t = h.dim(1)
        if t == 1 {
            return nil
        }
        // For multi-token, attention is over the sliding window plus the new t tokens; the
        // returned k has shape [..., min(maxSize, offset + t), ...]. Standard causal mask on
        // the query batch is correct because dropped tokens are not visible.
        let rinds = MLXArray(Int32(0)..<Int32(self.offset + t))
        let linds =
            self.offset != 0 ? MLXArray(Int32(self.offset)..<Int32(self.offset + t)) : rinds
        let mask = linds[0..., .newAxis] .< rinds[.newAxis]
        return (mask * Float32(-1e9)).asType(h.dtype)
    }
}
