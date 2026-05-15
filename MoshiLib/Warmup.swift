// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation

public enum WarmupMode: String, CaseIterable, Codable, Sendable {
    /// Run the standard warmup workload.
    case full
    /// Run a reduced warmup that still exercises the streaming path.
    /// `LM.warmup` already runs a single token so this is treated as `.full` there.
    case minimal
    /// Skip warmup entirely. First real step pays JIT/setup cost.
    case none
}
