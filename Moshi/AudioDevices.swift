import AVFoundation
import Foundation

#if os(macOS)
    import CoreAudio
#endif

struct AudioInputDevice: Identifiable, Hashable {
    // Empty string = "use system default"
    let uid: String
    let name: String
    var id: String { uid }

    static let systemDefault = AudioInputDevice(uid: "", name: "System default")

    static func list() -> [AudioInputDevice] {
        #if os(macOS)
            return macOSInputDevices()
        #elseif os(iOS)
            return iOSInputDevices()
        #else
            return []
        #endif
    }
}

#if os(macOS)
    // macOS-only Core Audio helpers. AudioDeviceID lookup is exposed so the
    // capture engine can apply the chosen device via AudioUnitSetProperty.
    enum MacAudio {
        static func deviceID(forUID uid: String) -> AudioDeviceID? {
            for (deviceUID, _, id) in enumerateInputDevices() where deviceUID == uid {
                return id
            }
            return nil
        }
    }

    private func macOSInputDevices() -> [AudioInputDevice] {
        enumerateInputDevices().map { AudioInputDevice(uid: $0.uid, name: $0.name) }
    }

    private func enumerateInputDevices() -> [(uid: String, name: String, id: AudioDeviceID)] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)

        var dataSize: UInt32 = 0
        var status = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &dataSize)
        guard status == noErr else { return [] }
        let count = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var ids = [AudioDeviceID](repeating: 0, count: count)
        status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &dataSize, &ids)
        guard status == noErr else { return [] }

        var devices: [(String, String, AudioDeviceID)] = []
        for id in ids {
            guard hasInputStreams(id: id) else { continue }
            guard let uid = deviceProperty(id: id, selector: kAudioDevicePropertyDeviceUID),
                let name = deviceProperty(id: id, selector: kAudioDevicePropertyDeviceNameCFString)
            else {
                continue
            }
            devices.append((uid, name, id))
        }
        return devices
    }

    private func hasInputStreams(id: AudioDeviceID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain)
        var size: UInt32 = 0
        let status = AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size)
        return status == noErr && size > 0
    }

    private func deviceProperty(id: AudioDeviceID, selector: AudioObjectPropertySelector)
        -> String?
    {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var cf: CFString? = nil
        var size = UInt32(MemoryLayout<CFString?>.size)
        let status = AudioObjectGetPropertyData(id, &address, 0, nil, &size, &cf)
        guard status == noErr, let cf else { return nil }
        return cf as String
    }
#endif

#if os(iOS)
    private func iOSInputDevices() -> [AudioInputDevice] {
        let session = AVAudioSession.sharedInstance()
        return (session.availableInputs ?? []).map {
            AudioInputDevice(uid: $0.uid, name: $0.portName)
        }
    }
#endif
