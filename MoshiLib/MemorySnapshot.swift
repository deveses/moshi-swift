// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import Darwin
import Foundation
import MLX

public struct MemorySnapshot: Codable, Sendable {
    public let label: String
    public let timestamp: TimeInterval
    public let residentBytes: Int
    public let mlxActiveBytes: Int
    public let mlxCacheBytes: Int
    public let mlxPeakBytes: Int
    public let kvCacheBytes: Int?

    public static func take(label: String, kvCacheBytes: Int? = nil) -> MemorySnapshot {
        MemorySnapshot(
            label: label,
            timestamp: Date().timeIntervalSince1970,
            residentBytes: residentMemoryBytes(),
            mlxActiveBytes: MLX.GPU.activeMemory,
            mlxCacheBytes: MLX.GPU.cacheMemory,
            mlxPeakBytes: MLX.GPU.peakMemory,
            kvCacheBytes: kvCacheBytes
        )
    }
}

public protocol MemorySink: AnyObject {
    func record(_ snapshot: MemorySnapshot)
}

public final class NoopMemorySink: MemorySink {
    public init() {}
    public func record(_ snapshot: MemorySnapshot) {}
}

public final class JSONLinesMemorySink: MemorySink, @unchecked Sendable {
    private let handle: FileHandle
    private let encoder: JSONEncoder
    private let lock = NSLock()

    public init(path: String) throws {
        let url = URL(fileURLWithPath: path)
        if !FileManager.default.fileExists(atPath: path) {
            FileManager.default.createFile(atPath: path, contents: nil)
        }
        self.handle = try FileHandle(forWritingTo: url)
        try self.handle.seekToEnd()
        self.encoder = JSONEncoder()
    }

    deinit {
        try? handle.close()
    }

    public func record(_ snapshot: MemorySnapshot) {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? encoder.encode(snapshot) else { return }
        handle.write(data)
        handle.write(Data([0x0A]))
    }
}

public final class MemoryLog: @unchecked Sendable {
    public static let shared = MemoryLog()
    private let lock = NSLock()
    private var _sink: MemorySink = NoopMemorySink()

    public var sink: MemorySink {
        get { lock.lock(); defer { lock.unlock() }; return _sink }
        set { lock.lock(); defer { lock.unlock() }; _sink = newValue }
    }

    public func snapshot(_ label: String, kvCacheBytes: Int? = nil) {
        sink.record(MemorySnapshot.take(label: label, kvCacheBytes: kvCacheBytes))
    }
}

private func residentMemoryBytes() -> Int {
    var info = mach_task_basic_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { intPtr in
            task_info(
                mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), intPtr, &count)
        }
    }
    return result == KERN_SUCCESS ? Int(info.resident_size) : -1
}
