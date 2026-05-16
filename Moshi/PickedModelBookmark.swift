import Foundation

enum PickedModelBookmark {
    private static let defaults = UserDefaults.standard

    private static func defaultsKey(for presetName: String) -> String {
        "PickedModelBookmark.\(presetName)"
    }

    static func save(url: URL, presetName: String) -> Bool {
        // Picker URLs need an active security scope before bookmarkData(.withSecurityScope)
        // will succeed. start/stop are balanced; safe to call here.
        let started = url.startAccessingSecurityScopedResource()
        defer {
            if started { url.stopAccessingSecurityScopedResource() }
        }
        do {
            #if os(macOS)
                let data = try url.bookmarkData(
                    options: [.withSecurityScope],
                    includingResourceValuesForKeys: nil,
                    relativeTo: nil)
            #else
                let data = try url.bookmarkData(
                    options: [],
                    includingResourceValuesForKeys: nil,
                    relativeTo: nil)
            #endif
            defaults.set(data, forKey: defaultsKey(for: presetName))
            return true
        } catch {
            print("PickedModelBookmark.save failed for \(presetName): \(error)")
            return false
        }
    }

    // Resolves the saved bookmark and begins security-scoped access. The scope is
    // held for the rest of the process because the loaded checkpoint is mmap'd
    // (release would invalidate the mapping).
    static func resolve(presetName: String) -> URL? {
        guard let data = defaults.data(forKey: defaultsKey(for: presetName)) else {
            return nil
        }
        var isStale = false
        do {
            #if os(macOS)
                let url = try URL(
                    resolvingBookmarkData: data,
                    options: [.withSecurityScope],
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale)
            #else
                let url = try URL(
                    resolvingBookmarkData: data,
                    options: [],
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale)
            #endif
            if isStale { return nil }
            _ = url.startAccessingSecurityScopedResource()
            return url
        } catch {
            print("PickedModelBookmark.resolve failed: \(error)")
            return nil
        }
    }
}
