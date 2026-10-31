#pragma once

#include "utils/common/type_common.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace awakening::runtime {

struct DaedalusPoseSnapshot {
    bool valid = false;
    uint64_t frame_sequence = 0;
    uint64_t timestamp_ns = 0;
    ISO3 pose = ISO3::Identity();
};

constexpr std::size_t DAEDALUS_SYNCHRONIZED_POSE_COUNT = 4;
using DaedalusFramePoseSnapshots =
    std::array<DaedalusPoseSnapshot, DAEDALUS_SYNCHRONIZED_POSE_COUNT>;

inline bool is_exact_daedalus_pose_bundle(
    const DaedalusFramePoseSnapshots& snapshots,
    uint64_t frame_sequence,
    uint64_t timestamp_ns
) noexcept {
    for (const auto& pose: snapshots) {
        if (!pose.valid || pose.frame_sequence != frame_sequence
            || pose.timestamp_ns != timestamp_ns)
        {
            return false;
        }
    }
    return true;
}

enum class DaedalusPoseBundleFailure : uint8_t {
    NONE = 0,
    MISSING_POSE,
    SEQUENCE_MISMATCH,
    TIMESTAMP_MISMATCH,
};

struct DaedalusPoseBundleInspection {
    DaedalusPoseBundleFailure failure = DaedalusPoseBundleFailure::MISSING_POSE;
    DaedalusFramePoseSnapshots poses {};

    [[nodiscard]] bool exact() const noexcept {
        return failure == DaedalusPoseBundleFailure::NONE;
    }
};

// The simulator callback writes poses while detector/tracker tasks read snapshots on other
// scheduler threads. This mutex is also the publication visibility boundary for a frame bundle.
class DaedalusFramePoseStore {
public:
    void record(
        std::size_t pose_index,
        uint64_t frame_sequence,
        uint64_t timestamp_ns,
        const ISO3& pose
    ) {
        if (pose_index >= DAEDALUS_SYNCHRONIZED_POSE_COUNT) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        auto [frame_it, inserted] = snapshots_.try_emplace(frame_sequence);
        if (inserted) {
            insertion_order_.push_back(frame_sequence);
        }
        auto& snapshot = frame_it->second[pose_index];
        snapshot.valid = true;
        snapshot.frame_sequence = frame_sequence;
        snapshot.timestamp_ns = timestamp_ns;
        snapshot.pose = pose;
        latest_[pose_index] = snapshot;
        while (snapshots_.size() > MAX_SNAPSHOTS) {
            snapshots_.erase(insertion_order_.front());
            insertion_order_.pop_front();
        }
    }

    [[nodiscard]] std::optional<DaedalusFramePoseSnapshots>
    exact(uint64_t frame_sequence, uint64_t timestamp_ns) const {
        const auto result = inspect(frame_sequence, timestamp_ns);
        if (!result.exact()) {
            return std::nullopt;
        }
        return result.poses;
    }

    [[nodiscard]] DaedalusPoseBundleInspection
    inspect(uint64_t frame_sequence, uint64_t timestamp_ns) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = snapshots_.find(frame_sequence);
        const auto poses = it == snapshots_.end() ? latest_ : it->second;
        for (const auto& pose: poses) {
            if (!pose.valid) {
                return { DaedalusPoseBundleFailure::MISSING_POSE, poses };
            }
        }
        for (const auto& pose: poses) {
            if (pose.frame_sequence != frame_sequence) {
                return { DaedalusPoseBundleFailure::SEQUENCE_MISMATCH, poses };
            }
        }
        for (const auto& pose: poses) {
            if (pose.timestamp_ns != timestamp_ns) {
                return { DaedalusPoseBundleFailure::TIMESTAMP_MISMATCH, poses };
            }
        }
        return { DaedalusPoseBundleFailure::NONE, poses };
    }

    [[nodiscard]] DaedalusFramePoseSnapshots snapshot(uint64_t frame_sequence) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = snapshots_.find(frame_sequence);
        return it == snapshots_.end() ? DaedalusFramePoseSnapshots {} : it->second;
    }

private:
    static constexpr std::size_t MAX_SNAPSHOTS = 1024;
    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, DaedalusFramePoseSnapshots> snapshots_;
    std::deque<uint64_t> insertion_order_;
    DaedalusFramePoseSnapshots latest_ {};
};

} // namespace awakening::runtime
