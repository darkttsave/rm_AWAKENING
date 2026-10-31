#include "runtime/daedalus_frame_sync.hpp"
#include "tasks/auto_buff/rune_track/motion_model.hpp"
#include "tasks/auto_buff/rune_track/rune_tracker.hpp"
#include "tasks/auto_buff/type.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <angles.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace {
using namespace awakening;
using namespace awakening::auto_buff;
namespace mm = awakening::auto_buff::motion_model;

constexpr double DT = 0.02;
constexpr int ODOM_FRAME_ID = 0;
const TimePoint BASE_TIME(std::chrono::seconds(100));

std::string tracker_state_name(RuneTarget::TrackState::State state) {
    return RuneTarget::TrackState::string_by_state(state);
}

const char* voter_state_name(int state) {
    switch (state) {
        case mm::Voter::Clockwise:
            return "Clockwise";
        case mm::Voter::Counterclockwise:
            return "Counterclockwise";
        default:
            return "Collecting";
    }
}

const char* voter_mode_name(int mode) {
    return mode == mm::Voter::Big ? "Big" : "Small";
}

struct ScenarioSpec {
    std::string name;
    int frames = 0;
    int observations = 1;
    double pixel_noise = 0.0;
    std::optional<std::pair<int, int>> missing_range;
    enum Motion { Static, ConstantPositive, ConstantNegative, SinusoidalPositive } motion = Static;
};

struct FrameMetric {
    std::string scenario;
    int frame = 0;
    double time_s = 0.0;
    double truth_roll = 0.0;
    double estimated_roll = 0.0;
    double estimated_v_roll = 0.0;
    double estimated_a = 0.0;
    double estimated_w = 0.0;
    double estimated_tau = 0.0;
    double pnp_center_error_m = 0.0;
    double tracker_center_error_m = 0.0;
    int requested_observations = 0;
    int pose_valid_observations = 0;
    int valid_matches = 0;
    int tracker_state = RuneTarget::TrackState::LOST;
    int voter_state = mm::Voter::Collecting;
    int voter_mode = mm::Voter::Small;
    int voter_count = 0;
    int double_detect_count = 0;
};

struct ScenarioResult {
    std::string name;
    bool pass = false;
    int frames = 0;
    int requested_observations = 0;
    int pose_valid_observations = 0;
    int valid_matches = 0;
    int pnp_outliers_gt_0p25m = 0;
    double pnp_error_max_m = 0.0;
    double tracker_error_after_warmup_mean_m = 0.0;
    double tracker_error_after_warmup_max_m = 0.0;
    double roll_error_after_warmup_mean_rad = 0.0;
    double roll_error_after_warmup_max_rad = 0.0;
    bool saw_tracking = false;
    bool saw_temp_lost = false;
    bool recovered_tracking = false;
    bool saw_lost_after_tracking = false;
    bool saw_voter_reset_on_reinit = false;
    int final_voter_state = mm::Voter::Collecting;
    int final_voter_mode = mm::Voter::Small;
    int final_voter_count = 0;
    int final_double_detect_count = 0;
    double final_a = 0.0;
    double final_w = 0.0;
    std::string note;
};

CameraInfo make_camera_info() {
    constexpr double focal = 1303.6753236814714;
    return CameraInfo {
        .camera_matrix =
            (cv::Mat_<double>(3, 3) << focal, 0.0, 720.0, 0.0, focal, 540.0, 0.0, 0.0, 1.0),
        .distortion_coefficients = cv::Mat::zeros(1, 5, CV_64F),
    };
}

ISO3 make_camera_pose() {
    ISO3 camera_cv_in_odom = ISO3::Identity();
    camera_cv_in_odom.translation() = Vec3(0.25, -0.10, 0.40);
    return camera_cv_in_odom;
}

double truth_roll(const ScenarioSpec& spec, double t) {
    constexpr double initial_roll = 0.20;
    switch (spec.motion) {
        case ScenarioSpec::ConstantPositive:
            return initial_roll + mm::SMALL_SPEED * t;
        case ScenarioSpec::ConstantNegative:
            return initial_roll - mm::SMALL_SPEED * t;
        case ScenarioSpec::SinusoidalPositive: {
            constexpr double a = 0.90;
            constexpr double w = 1.92;
            constexpr double b = mm::AMPLITUDE_SUM - a;
            return initial_roll + (a / w) * (1.0 - std::cos(w * t)) + b * t;
        }
        default:
            return initial_roll;
    }
}

mm::VecX make_truth_state(double roll) {
    mm::VecX state = mm::VecX::Zero();
    state[mm::idx::CX] = 0.25;
    state[mm::idx::CY] = -0.10;
    state[mm::idx::CZ] = 6.40;
    state[mm::idx::ROLL] = mm::normalize_angle(roll);
    state[mm::idx::V_ROLL] = 0.0;
    state[mm::idx::A] = 0.90;
    state[mm::idx::W] = 1.92;
    return state;
}

RuneFanBladeWithR make_fan(
    const mm::VecX& truth,
    int blade_id,
    int frame,
    double noise_px,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    const ISO3 pose_in_odom = mm::rune_pose(truth.data(), blade_id);
    const ISO3 pose_in_camera = camera_cv_in_odom.inverse() * pose_in_odom;
    const auto object_points =
        RuneFanBladeWithR::Point3DRZERO<cv::Point3f>::build_cb5_five_points();
    auto image_points = utils::reprojection(
        camera_info.camera_matrix,
        camera_info.distortion_coefficients,
        object_points,
        pose_in_camera
    );
    for (std::size_t point = 0; point < image_points.size(); ++point) {
        const double phase = frame * 0.731 + static_cast<double>(point) * 1.913
            + static_cast<double>(blade_id) * 0.417;
        image_points[point].x += static_cast<float>(noise_px * std::sin(phase));
        image_points[point].y += static_cast<float>(noise_px * std::cos(phase * 1.173));
    }

    RuneFanBladeWithR fan;
    fan.geometry = RuneBladeGeometry::CB5_EDGE_CENTERS_R150;
    fan.activation = RuneActivation::INACTIVE;
    fan.confidence = 1.0;
    fan.points.resize(RuneFanBladeWithR::N);
    fan.points[RuneFanBladeWithR::TOP] = image_points[0];
    fan.points[RuneFanBladeWithR::LEFT] = image_points[1];
    fan.points[RuneFanBladeWithR::BOTTOM] = image_points[2];
    fan.points[RuneFanBladeWithR::RIGHT] = image_points[3];
    fan.points[RuneFanBladeWithR::CENTER] =
        (image_points[0] + image_points[1] + image_points[2] + image_points[3]) * 0.25F;
    fan.points[RuneFanBladeWithR::R] = image_points[4];
    fan.raw_points = {
        image_points[0], image_points[1], image_points[4], image_points[3], image_points[2]
    };
    fan.raw_confidences.fill(1.0F);
    return fan;
}

RuneDetection make_detection(
    const ScenarioSpec& spec,
    int frame,
    double roll,
    const CameraInfo& camera_info,
    const ISO3& camera_cv_in_odom
) {
    RuneDetection detection;
    detection.id = frame;
    detection.frame_id = ODOM_FRAME_ID;
    detection.timestamp = BASE_TIME
        + std::chrono::duration_cast<TimePoint::duration>(std::chrono::duration<double>(frame * DT));
    if (spec.missing_range && frame >= spec.missing_range->first
        && frame < spec.missing_range->second)
    {
        return detection;
    }
    const auto truth = make_truth_state(roll);
    for (int blade_id = 0; blade_id < spec.observations; ++blade_id) {
        detection.fan_blades.push_back(
            make_fan(
                truth,
                blade_id,
                frame,
                spec.pixel_noise,
                camera_info,
                camera_cv_in_odom
            )
        );
    }
    return detection;
}

ScenarioResult run_scenario(
    const ScenarioSpec& spec,
    const YAML::Node& tracker_config,
    std::vector<FrameMetric>& all_metrics
) {
    RuneTracker tracker(tracker_config);
    const auto camera_info = make_camera_info();
    const auto camera_pose = make_camera_pose();
    const Vec3 truth_center(0.25, -0.10, 6.40);
    ScenarioResult result;
    result.name = spec.name;
    result.frames = spec.frames;
    double tracker_error_sum = 0.0;
    double roll_error_sum = 0.0;
    int tracker_error_count = 0;
    bool was_temp_lost = false;
    bool was_tracking = false;

    for (int frame = 0; frame < spec.frames; ++frame) {
        const double time_s = frame * DT;
        const double roll = truth_roll(spec, time_s);
        auto detection = make_detection(spec, frame, roll, camera_info, camera_pose);
        result.requested_observations += static_cast<int>(detection.fan_blades.size());
        auto target = tracker.track(detection, camera_info, camera_pose, ODOM_FRAME_ID);
        const auto diagnostics = tracker.diagnostics();
        result.pose_valid_observations += diagnostics.pose_valid_fan_blades;
        result.valid_matches += diagnostics.valid_matches;
        result.saw_tracking = result.saw_tracking
            || target.track_state.tracker_state == RuneTarget::TrackState::TRACKING;
        was_tracking = was_tracking
            || target.track_state.tracker_state == RuneTarget::TrackState::TRACKING;
        if (was_tracking && target.track_state.tracker_state == RuneTarget::TrackState::LOST) {
            result.saw_lost_after_tracking = true;
        }
        if (result.saw_lost_after_tracking && diagnostics.initialized
            && target.voter.state == mm::Voter::Collecting
            && target.voter.clock_wise_count == 0)
        {
            result.saw_voter_reset_on_reinit = true;
        }
        if (target.track_state.tracker_state == RuneTarget::TrackState::TEMP_LOST) {
            result.saw_temp_lost = true;
            was_temp_lost = true;
        }
        if (was_temp_lost && target.track_state.tracker_state == RuneTarget::TrackState::TRACKING) {
            result.recovered_tracking = true;
        }

        double pnp_error = 0.0;
        for (const auto& fan: detection.fan_blades) {
            if (fan.pose_valid) {
                pnp_error = std::max(pnp_error, (fan.pose.translation() - truth_center).norm());
            }
        }
        result.pnp_error_max_m = std::max(result.pnp_error_max_m, pnp_error);
        result.pnp_outliers_gt_0p25m += pnp_error > 0.25;
        const double tracker_error = (target.get_target_state().pos() - truth_center).norm();
        if (frame >= 20 && target.track_state.tracker_state != RuneTarget::TrackState::LOST) {
            tracker_error_sum += tracker_error;
            const double roll_error = std::abs(
                angles::normalize_angle(target.get_target_state().roll() - roll)
            );
            roll_error_sum += roll_error;
            ++tracker_error_count;
            result.tracker_error_after_warmup_max_m =
                std::max(result.tracker_error_after_warmup_max_m, tracker_error);
            result.roll_error_after_warmup_max_rad =
                std::max(result.roll_error_after_warmup_max_rad, roll_error);
        }
        all_metrics.push_back(
            FrameMetric {
                .scenario = spec.name,
                .frame = frame,
                .time_s = time_s,
                .truth_roll = mm::normalize_angle(roll),
                .estimated_roll = target.get_target_state().roll(),
                .estimated_v_roll = target.get_target_state().v_roll(target.voter),
                .estimated_a = target.get_target_state().a(),
                .estimated_w = target.get_target_state().w(),
                .estimated_tau = target.get_target_state().tau(),
                .pnp_center_error_m = pnp_error,
                .tracker_center_error_m = tracker_error,
                .requested_observations = static_cast<int>(detection.fan_blades.size()),
                .pose_valid_observations = diagnostics.pose_valid_fan_blades,
                .valid_matches = diagnostics.valid_matches,
                .tracker_state = target.track_state.tracker_state,
                .voter_state = target.voter.state,
                .voter_mode = target.voter.mode,
                .voter_count = target.voter.clock_wise_count,
                .double_detect_count = target.voter.double_detect_count,
            }
        );
        result.final_voter_state = target.voter.state;
        result.final_voter_mode = target.voter.mode;
        result.final_voter_count = target.voter.clock_wise_count;
        result.final_double_detect_count = target.voter.double_detect_count;
        result.final_a = target.get_target_state().a();
        result.final_w = target.get_target_state().w();
    }

    if (tracker_error_count > 0) {
        result.tracker_error_after_warmup_mean_m = tracker_error_sum / tracker_error_count;
        result.roll_error_after_warmup_mean_rad = roll_error_sum / tracker_error_count;
    }
    // RuneTarget::reset stops after the first valid fan, so extra observations in the
    // initialization frame are intentionally not PnP-processed by the production path.
    const int expected_pose_valid =
        result.requested_observations - std::max(0, spec.observations - 1);
    const bool pnp_condition = spec.pixel_noise > 0.0
        ? result.pose_valid_observations == expected_pose_valid
            && result.valid_matches >= static_cast<int>(0.90 * (spec.frames - 1))
        : result.pose_valid_observations == expected_pose_valid
            && result.pnp_error_max_m < 1e-3;
    const bool common = pnp_condition && result.tracker_error_after_warmup_max_m < 0.25
        && result.saw_tracking;
    bool scenario_condition = true;
    if (spec.motion == ScenarioSpec::ConstantPositive) {
        scenario_condition = result.final_voter_state == mm::Voter::Clockwise;
    } else if (spec.motion == ScenarioSpec::ConstantNegative) {
        scenario_condition = result.final_voter_state == mm::Voter::Counterclockwise;
    } else if (spec.motion == ScenarioSpec::SinusoidalPositive) {
        scenario_condition = result.final_voter_state == mm::Voter::Clockwise
            && result.final_voter_mode == mm::Voter::Big;
    }
    if (spec.missing_range) {
        scenario_condition = scenario_condition && result.saw_temp_lost
            && result.recovered_tracking;
    }
    if (spec.name == "long_missing_reinitialize") {
        scenario_condition = scenario_condition && result.saw_lost_after_tracking
            && result.saw_voter_reset_on_reinit;
    }
    if (spec.observations == 1 && spec.motion != ScenarioSpec::SinusoidalPositive) {
        scenario_condition = scenario_condition && result.final_voter_mode == mm::Voter::Small;
    }
    result.pass = common && scenario_condition;
    result.note = result.pass ? "stable" : "see frame_metrics.csv";
    return result;
}

bool test_pose_visibility(std::string& detail) {
    // Deterministic reproduction of the legacy ordering: an immediately scheduled consumer
    // runs after image publication but before any pose is imported.
    int legacy_missing = 0;
    {
        runtime::DaedalusFramePoseStore legacy_store;
        for (uint64_t sequence = 1; sequence <= 500; ++sequence) {
            const uint64_t timestamp = 8'000'000'000ULL + sequence * 1'000'000ULL;
            legacy_missing += !legacy_store.exact(sequence, timestamp);
            for (std::size_t pose_index = 0;
                 pose_index < runtime::DAEDALUS_SYNCHRONIZED_POSE_COUNT;
                 ++pose_index)
            {
                legacy_store.record(pose_index, sequence, timestamp, ISO3::Identity());
            }
        }
    }
    runtime::DaedalusFramePoseStore store;
    constexpr int frame_count = 500;
    constexpr uint64_t base_timestamp = 9'000'000'000ULL;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<uint64_t> published;
    bool producer_done = false;
    std::atomic<bool> ok { true };

    std::thread consumer([&]() {
        int consumed = 0;
        while (consumed < frame_count) {
            uint64_t sequence = 0;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait(lock, [&]() { return !published.empty() || producer_done; });
                if (published.empty()) {
                    break;
                }
                sequence = published.front();
                published.pop_front();
            }
            const uint64_t timestamp = base_timestamp + sequence * 1'000'000ULL;
            const auto exact = store.exact(sequence, timestamp);
            if (!exact) {
                ok = false;
                break;
            }
            for (std::size_t pose_index = 0;
                 pose_index < runtime::DAEDALUS_SYNCHRONIZED_POSE_COUNT;
                 ++pose_index)
            {
                const double expected = static_cast<double>(sequence * 10 + pose_index);
                if (std::abs((*exact)[pose_index].pose.translation().x() - expected) > 1e-12) {
                    ok = false;
                }
            }
            ++consumed;
        }
        if (consumed != frame_count) {
            ok = false;
        }
    });

    for (uint64_t sequence = 1; sequence <= frame_count; ++sequence) {
        const uint64_t timestamp = base_timestamp + sequence * 1'000'000ULL;
        for (std::size_t pose_index = 0;
             pose_index < runtime::DAEDALUS_SYNCHRONIZED_POSE_COUNT;
             ++pose_index)
        {
            ISO3 pose = ISO3::Identity();
            pose.translation().x() = static_cast<double>(sequence * 10 + pose_index);
            store.record(pose_index, sequence, timestamp, pose);
            if (pose_index + 1 < runtime::DAEDALUS_SYNCHRONIZED_POSE_COUNT
                && store.exact(sequence, timestamp))
            {
                ok = false;
            }
        }
        if (!store.exact(sequence, timestamp) || store.exact(sequence, timestamp + 1)) {
            ok = false;
        }
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            published.push_back(sequence);
        }
        queue_cv.notify_one();
    }
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        producer_done = true;
    }
    queue_cv.notify_all();
    consumer.join();
    std::unordered_map<uint64_t, runtime::DaedalusFramePoseSnapshots> legacy_retention;
    int legacy_retention_drops = 0;
    constexpr uint64_t retention_frames = 4096;
    for (uint64_t sequence = 1; sequence <= retention_frames; ++sequence) {
        const uint64_t timestamp = base_timestamp + sequence * 1'000'000ULL;
        for (std::size_t pose_index = 0;
             pose_index < runtime::DAEDALUS_SYNCHRONIZED_POSE_COUNT;
             ++pose_index)
        {
            auto& pose = legacy_retention[sequence][pose_index];
            pose.valid = true;
            pose.frame_sequence = sequence;
            pose.timestamp_ns = timestamp;
            if (legacy_retention.size() > 1024) {
                legacy_retention.erase(legacy_retention.begin());
            }
        }
        const auto it = legacy_retention.find(sequence);
        legacy_retention_drops += it == legacy_retention.end()
            || !runtime::is_exact_daedalus_pose_bundle(it->second, sequence, timestamp);
    }
    runtime::DaedalusFramePoseStore retention_store;
    bool retention_ok = true;
    for (uint64_t sequence = 1; sequence <= retention_frames; ++sequence) {
        const uint64_t timestamp = base_timestamp + sequence * 1'000'000ULL;
        for (std::size_t pose_index = 0;
             pose_index < runtime::DAEDALUS_SYNCHRONIZED_POSE_COUNT;
             ++pose_index)
        {
            retention_store.record(pose_index, sequence, timestamp, ISO3::Identity());
        }
        retention_ok = retention_ok && retention_store.exact(sequence, timestamp).has_value();
    }
    const auto timestamp_failure = retention_store.inspect(
        retention_frames,
        base_timestamp + retention_frames * 1'000'000ULL + 1
    );
    const auto sequence_failure = retention_store.inspect(
        retention_frames + 1,
        base_timestamp + (retention_frames + 1) * 1'000'000ULL
    );
    const bool classified = timestamp_failure.failure
            == runtime::DaedalusPoseBundleFailure::TIMESTAMP_MISMATCH
        && sequence_failure.failure
            == runtime::DaedalusPoseBundleFailure::SEQUENCE_MISMATCH;
    detail = "legacy_missing=" + std::to_string(legacy_missing)
        + "/500; fixed_exact_visible=" + std::to_string(ok ? 500 : 0)
        + "/500; legacy_eviction_drops=" + std::to_string(legacy_retention_drops)
        + "/" + std::to_string(retention_frames)
        + "; retention_exact=" + std::to_string(retention_ok ? retention_frames : 0)
        + "/" + std::to_string(retention_frames)
        + "; failure_classification=" + (classified ? "PASS" : "FAIL");
    return ok && legacy_missing == 500 && legacy_retention_drops > 0 && retention_ok
        && classified;
}

void write_metrics(
    const std::filesystem::path& output_dir,
    const std::vector<FrameMetric>& metrics,
    const std::vector<ScenarioResult>& results,
    bool pose_sync_pass,
    const std::string& pose_sync_detail
) {
    std::filesystem::create_directories(output_dir);
    std::ofstream frame_file(output_dir / "frame_metrics.csv");
    frame_file << std::setprecision(17)
               << "scenario,frame,time_s,truth_roll,estimated_roll,estimated_v_roll,"
                  "estimated_a,estimated_w,estimated_tau,pnp_center_error_m,"
                  "tracker_center_error_m,requested_observations,pose_valid_observations,"
                  "valid_matches,tracker_state,voter_state,voter_mode,voter_count,"
                  "double_detect_count\n";
    for (const auto& row: metrics) {
        frame_file << row.scenario << ',' << row.frame << ',' << row.time_s << ','
                   << row.truth_roll << ',' << row.estimated_roll << ','
                   << row.estimated_v_roll << ',' << row.estimated_a << ',' << row.estimated_w
                   << ',' << row.estimated_tau << ','
                   << row.pnp_center_error_m << ',' << row.tracker_center_error_m << ','
                   << row.requested_observations << ',' << row.pose_valid_observations << ','
                   << row.valid_matches << ',' << tracker_state_name(
                          static_cast<RuneTarget::TrackState::State>(row.tracker_state)
                      )
                   << ',' << voter_state_name(row.voter_state) << ','
                   << voter_mode_name(row.voter_mode) << ',' << row.voter_count << ','
                   << row.double_detect_count << '\n';
    }

    std::ofstream summary_file(output_dir / "scenario_summary.csv");
    summary_file << std::setprecision(17)
                 << "scenario,pass,frames,requested_observations,pose_valid_observations,"
                    "valid_matches,pnp_outliers_gt_0p25m,pnp_error_max_m,"
                    "tracker_error_after_warmup_mean_m,"
                    "tracker_error_after_warmup_max_m,roll_error_after_warmup_mean_rad,"
                    "roll_error_after_warmup_max_rad,saw_tracking,saw_temp_lost,"
                    "recovered_tracking,saw_lost_after_tracking,saw_voter_reset_on_reinit,"
                    "final_voter_state,final_voter_mode,final_voter_count,"
                    "final_double_detect_count,final_a,final_w,note\n";
    for (const auto& row: results) {
        summary_file << row.name << ',' << row.pass << ',' << row.frames << ','
                     << row.requested_observations << ',' << row.pose_valid_observations << ','
                     << row.valid_matches << ',' << row.pnp_outliers_gt_0p25m << ','
                     << row.pnp_error_max_m << ','
                     << row.tracker_error_after_warmup_mean_m << ','
                     << row.tracker_error_after_warmup_max_m << ','
                     << row.roll_error_after_warmup_mean_rad << ','
                     << row.roll_error_after_warmup_max_rad << ',' << row.saw_tracking << ','
                     << row.saw_temp_lost << ',' << row.recovered_tracking << ','
                     << row.saw_lost_after_tracking << ',' << row.saw_voter_reset_on_reinit << ','
                     << voter_state_name(row.final_voter_state) << ','
                     << voter_mode_name(row.final_voter_mode) << ',' << row.final_voter_count
                     << ',' << row.final_double_detect_count << ',' << row.final_a << ','
                     << row.final_w << ',' << row.note << '\n';
    }
    std::ofstream transition_file(output_dir / "state_transitions.csv");
    transition_file << "scenario,frame,time_s,machine,from_state,to_state\n";
    const FrameMetric* previous = nullptr;
    for (const auto& row: metrics) {
        const bool new_scenario = previous == nullptr || previous->scenario != row.scenario;
        if (new_scenario) {
            transition_file << row.scenario << ',' << row.frame << ',' << row.time_s
                            << ",tracker,START,"
                            << tracker_state_name(
                                   static_cast<RuneTarget::TrackState::State>(row.tracker_state)
                               )
                            << '\n';
            transition_file << row.scenario << ',' << row.frame << ',' << row.time_s
                            << ",voter_state,START," << voter_state_name(row.voter_state) << '\n';
            transition_file << row.scenario << ',' << row.frame << ',' << row.time_s
                            << ",voter_mode,START," << voter_mode_name(row.voter_mode) << '\n';
        } else {
            if (previous->tracker_state != row.tracker_state) {
                transition_file << row.scenario << ',' << row.frame << ',' << row.time_s
                                << ",tracker,"
                                << tracker_state_name(static_cast<RuneTarget::TrackState::State>(
                                       previous->tracker_state
                                   ))
                                << ','
                                << tracker_state_name(static_cast<RuneTarget::TrackState::State>(
                                       row.tracker_state
                                   ))
                                << '\n';
            }
            if (previous->voter_state != row.voter_state) {
                transition_file << row.scenario << ',' << row.frame << ',' << row.time_s
                                << ",voter_state," << voter_state_name(previous->voter_state)
                                << ',' << voter_state_name(row.voter_state) << '\n';
            }
            if (previous->voter_mode != row.voter_mode) {
                transition_file << row.scenario << ',' << row.frame << ',' << row.time_s
                                << ",voter_mode," << voter_mode_name(previous->voter_mode) << ','
                                << voter_mode_name(row.voter_mode) << '\n';
            }
        }
        previous = &row;
    }
    std::ofstream sync_file(output_dir / "pose_sync_test.txt");
    sync_file << (pose_sync_pass ? "PASS" : "FAIL") << ": " << pose_sync_detail << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: " << argv[0] << " <config.yaml> [output-dir]\n";
        return 2;
    }
    awakening::logger::init(spdlog::level::warn);
    const YAML::Node config = YAML::LoadFile(argv[1]);
    const auto tracker_config = config["rune_tracker"];
    const std::filesystem::path output_dir = argc == 3
        ? std::filesystem::path(argv[2])
        : std::filesystem::path("ai_workspace/artifacts/task-r4-downstream-20261030");

    std::string pose_sync_detail;
    const bool pose_sync_pass = test_pose_visibility(pose_sync_detail);
    std::vector<ScenarioSpec> specs {
        { .name = "static_single", .frames = 120, .observations = 1,
          .motion = ScenarioSpec::Static },
        { .name = "constant_clockwise_single", .frames = 300, .observations = 1,
          .motion = ScenarioSpec::ConstantPositive },
        { .name = "constant_counterclockwise_single", .frames = 300, .observations = 1,
          .motion = ScenarioSpec::ConstantNegative },
        { .name = "constant_clockwise_double", .frames = 300, .observations = 2,
          .motion = ScenarioSpec::ConstantPositive },
        { .name = "sinusoidal_big_double", .frames = 350, .observations = 2,
          .motion = ScenarioSpec::SinusoidalPositive },
        { .name = "short_missing_recovery", .frames = 220, .observations = 1,
          .missing_range = std::pair { 90, 100 },
          .motion = ScenarioSpec::ConstantPositive },
        { .name = "long_missing_reinitialize", .frames = 450, .observations = 1,
          .missing_range = std::pair { 180, 250 },
          .motion = ScenarioSpec::ConstantPositive },
        { .name = "controlled_noise_0p15px", .frames = 300, .observations = 1,
          .pixel_noise = 0.15, .motion = ScenarioSpec::ConstantPositive },
    };

    std::vector<FrameMetric> metrics;
    std::vector<ScenarioResult> results;
    bool scenarios_pass = true;
    for (const auto& spec: specs) {
        auto result = run_scenario(spec, tracker_config, metrics);
        scenarios_pass = scenarios_pass && result.pass;
        std::cout << result.name << " " << (result.pass ? "PASS" : "FAIL")
                  << " pnp_max_m=" << result.pnp_error_max_m
                  << " tracker_mean/max_m=" << result.tracker_error_after_warmup_mean_m << "/"
                  << result.tracker_error_after_warmup_max_m << " matches="
                  << result.valid_matches << " voter="
                  << voter_state_name(result.final_voter_state) << "/"
                  << voter_mode_name(result.final_voter_mode) << "\n";
        results.push_back(std::move(result));
    }
    write_metrics(output_dir, metrics, results, pose_sync_pass, pose_sync_detail);
    std::cout << "pose_sync " << (pose_sync_pass ? "PASS" : "FAIL") << " "
              << pose_sync_detail << "\n";
    const bool pass = pose_sync_pass && scenarios_pass;
    std::cout << (pass ? "DOWNSTREAM_TEST PASS" : "DOWNSTREAM_TEST FAIL") << "\n";
    return pass ? 0 : 1;
}
