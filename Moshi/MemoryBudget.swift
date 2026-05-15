// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Foundation

enum MemoryBudget {
    /// Approximate per-process budget available for a model load.
    /// On macOS we reserve 2 GB for the OS and other apps; on iOS we apply a
    /// jetsam-aware heuristic (no API to read the real limit).
    /// Debug builds honour `MEMORY_BUDGET_OVERRIDE_BYTES` for manual testing.
    static func availableBytes() -> Int {
        if let override = ProcessInfo.processInfo.environment["MEMORY_BUDGET_OVERRIDE_BYTES"],
            let bytes = Int(override)
        {
            return bytes
        }
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        #if os(iOS)
            return min(Int(Double(physical) * 0.4), 4 * 1024 * 1024 * 1024)
        #else
            return max(physical - 2 * 1024 * 1024 * 1024, 0)
        #endif
    }

    static func fitsLikely(estimatedBytes: Int) -> Bool {
        estimatedBytes <= availableBytes()
    }
}

extension Int {
    var formattedGB: String {
        String(format: "%.1f GB", Double(self) / 1_073_741_824.0)
    }
}
