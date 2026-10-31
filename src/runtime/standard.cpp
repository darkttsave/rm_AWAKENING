#include "angles.h"
#include "ascii_banner.hpp"
#include "daedalus_interface/shm_layout.hpp"
#include "runtime/daedalus_frame_sync.hpp"
#include "tasks/auto_aim/armor_track/motion_model.hpp"
#include "tasks/auto_buff/rune_control/rune_aimer.hpp"
#include "tasks/auto_buff/rune_detect/rune_detector.hpp"
#include "tasks/auto_buff/rune_track/rune_target.hpp"
#include "tasks/auto_buff/rune_track/rune_tracker.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/base/ballistic_trajectory.hpp"
#include "tasks/base/wheel_odometry.hpp"
#include "utils/drivers/camera_factory.hpp"
#include "utils/io/video_save.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <opencv2/highgui.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#ifdef USE_RERUN
    #include "_rerun/recorder.hpp"
    #include "_rerun/tf.hpp"
#endif
#ifdef USE_ROS2
    #include "_rcl/node.hpp"
    #include "_rcl/tf.hpp"
    #include "_rcl/visual/armor.hpp"
    #include "_rcl/visual/armor_target.hpp"
    #include "sensor_msgs/msg/camera_info.hpp"
    #include "sensor_msgs/msg/image.hpp"
    #include <rclcpp/qos.hpp>
#endif
#include "backward-cpp/backward.hpp"
#ifdef USE_Daedalus
    #include "daedalus_interface/shm_client.hpp"
#endif
#include "param_deliver.h"
#include "runtime/config.hpp"
#include "tasks/auto_aim/armor_control/very_aimer.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/auto_aim/armor_track/armor_target.hpp"
#include "tasks/auto_aim/armor_track/armor_tracker.hpp"
#include "tasks/auto_aim/auto_aim_fsm.hpp"
#include "tasks/auto_aim/type.hpp"
#include "tasks/base/common.hpp"
#include "tasks/base/packet_typedef_receive.hpp"
#include "tasks/base/packet_typedef_send.hpp"
#include "tasks/base/web.hpp"
#include "utils/buffer.hpp"
#include "utils/common/image.hpp"
#include "utils/common/type_common.hpp"
#include "utils/drivers/serial_driver.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_tf.hpp"
#include "utils/scheduler/scheduler.hpp"
#include "utils/semaphore_guard.hpp"
#include "utils/signal_guard.hpp"
#include "utils/utils.hpp"
namespace backward {
static backward::SignalHandling sh;
}
using namespace awakening;
enum class Mode : int { AutoAim = 0, AutoBuff = 1 };
Mode str_to_Mode(std::string str) {
    str = utils::to_upper(str);
    if (str == "AUTO_AIM") {
        return Mode::AutoAim;
    } else if (str == "AUTO_BUFF") {
        return Mode::AutoBuff;
    } else {
        throw std::invalid_argument("Invalid mode string");
    }
}
enum class SimpleFrame : int { ODOM, GIMBAL_ODOM, GIMBAL, CAMERA, CAMERA_CV, SHOOT, N };

using SimpleRobotTF = utils::tf::RobotTF<SimpleFrame, static_cast<size_t>(SimpleFrame::N), false>;

std::string SimpleFrame_to_str(int frame) {
    constexpr const char* details[] = { "odom",   "gimbal_odom", "gimbal",
                                        "camera", "camera_cv",   "shoot" };
    return std::string(details[frame]);
}
std::string SimpleFrame_to_str(SimpleFrame frame) {
    return SimpleFrame_to_str(std::to_underlying(frame));
}
struct CameraTag {};
struct SerialTag {};
struct DetectTag {};
struct FrameTag {};

using CameraIO = IOPair<CameraTag, ImageFrame>;
using SerialIO = IOPair<SerialTag, std::vector<uint8_t>>;
using CommonFrameIo = IOPair<FrameTag, CommonFrame>;
struct TimedRuneDetection {
    auto_buff::RuneDetection detection;
    int id = -1;
    bool detected = false;
    TimePoint detector_start {};
    TimePoint detector_end {};
    cv::Rect focus;
    bool full_frame = true;
    uint64_t source_sequence = 0;
};
struct Detection {
    std::optional<std::vector<auto_aim::Armors>> armors;
    std::optional<std::vector<TimedRuneDetection>> rune;
    std::optional<TimePoint> detect_start;
};
using DetIo = IOPair<DetectTag, Detection>;
struct LogCtx {
    int camera_count = 0;
    int detect_count = 0;
    int track_count = 0;
    int solve_count = 0;
    int serial_count = 0;
    int found_count = 0;
    double latency_ms_total = 0.0;
    void reset() {
        *this = {};
    }
};
struct Cb5PerfSample {
    bool detected = false;
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double output_copy_ms = 0.0;
    double postprocess_ms = 0.0;
    double traditional_cv_ms = 0.0;
    double detector_ms = 0.0;
    double queue_wait_ms = 0.0;
    double tracker_ms = 0.0;
    double total_ms = 0.0;
    cv::Rect focus;
    bool full_frame = true;
    int fan_blades = 0;
    int pose_valid_fan_blades = 0;
    int valid_matches = 0;
    bool initialized = false;
    auto_buff::RuneTarget::TrackState::State tracker_state =
        auto_buff::RuneTarget::TrackState::LOST;
    int clock_wise_count = 0;
    int double_detect_count = 0;
    int voter_state = auto_buff::motion_model::Voter::Collecting;
    int voter_mode = auto_buff::motion_model::Voter::Small;
};
struct Cb5PerfCtx {
    void add(Cb5PerfSample sample) {
        std::lock_guard<std::mutex> lock(mutex);
        samples.push_back(sample);
    }
    std::vector<Cb5PerfSample> take() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<Cb5PerfSample> result;
        result.swap(samples);
        return result;
    }
    std::mutex mutex;
    std::vector<Cb5PerfSample> samples;
};

struct Cb5Capture {
    explicit Cb5Capture(std::filesystem::path root, int frame_limit):
        root(std::move(root)), frame_limit(frame_limit) {
        std::filesystem::create_directories(this->root / "images");
        frames.open(this->root / "frames.csv");
        candidates.open(this->root / "candidates.csv");
        fans.open(this->root / "fans.csv");
        ground_truth.open(this->root / "ground_truth.csv");
        frames
            << "frame_id,source_seq,image_timestamp_ns,detected,raw_anchors,class_candidates,"
               "reject_min_valid,reject_invalid_r,reject_invalid_cardinal,reject_aw_strict,"
               "aw_before_nms,aw_after_nms,original_before_nms,original_after_nms,final_fans,"
               "pose_sync_exact,camera_pose_seq,camera_pose_timestamp_ns,gimbal_pose_seq,"
               "gimbal_pose_timestamp_ns,odom_pose_seq,odom_pose_timestamp_ns,muzzle_pose_seq,"
               "muzzle_pose_timestamp_ns,camera_odom_x,camera_odom_y,camera_odom_z,"
               "camera_odom_qw,camera_odom_qx,camera_odom_qy,camera_odom_qz,"
               "camera_fx,camera_fy,camera_cx,camera_cy,camera_d0,camera_d1,camera_d2,"
               "camera_d3,camera_d4,image_width,image_height,"
               "tracker_x,tracker_y,tracker_z,tracker_state,valid_matches\n";
        candidates
            << "frame_id,source_seq,anchor,class_id,class_confidence,valid_keypoints,"
               "reject_min_valid,reject_invalid_r,reject_invalid_cardinal,accepted_aw,"
               "accepted_original,p0_x,p0_y,p0_conf,p1_x,p1_y,p1_conf,p2_x,p2_y,p2_conf,"
               "p3_x,p3_y,p3_conf,p4_x,p4_y,p4_conf\n";
        ground_truth
            << "frame_id,source_seq,image_timestamp_ns,gt_present,gt_sync_exact,gt_batch_seq,"
               "gt_batch_timestamp_ns,rune_index,rune_frame_seq,rune_timestamp_ns,team,"
               "rune_mode,mechanism_state,r_center_odom_x,r_center_odom_y,r_center_odom_z,"
               "radius,current_angle,v_roll,direction,sin_amplitude,sin_omega,sin_phase,"
               "sin_offset,relative_time,blade_id,target_0,target_1,target_2,target_3,"
               "target_4\n";
        fans
            << "frame_id,source_seq,fan_index,activation,class_confidence,pose_valid,"
               "reprojection_error_px,camera_x,camera_y,camera_z,odom_x,odom_y,odom_z,"
               "predicted_x,predicted_y,predicted_z,best_blade_id,matched_blade_id,best_d2,"
               "residual_yaw,residual_pitch,residual_distance,residual_rot_yaw,"
               "residual_rot_roll,p0_x,p0_y,p0_conf,p1_x,p1_y,p1_conf,p2_x,p2_y,p2_conf,"
               "p3_x,p3_y,p3_conf,p4_x,p4_y,p4_conf\n";
        frames << std::setprecision(17);
        candidates << std::setprecision(9);
        fans << std::setprecision(17);
        ground_truth << std::setprecision(17);
    }

    bool includes(int frame_id) const noexcept {
        return frame_id >= 0 && frame_id < frame_limit;
    }

    void record_ground_truth(const talos::ipc::GroundTruthBatch& batch) {
        std::lock_guard<std::mutex> lock(mutex);
        ground_truth_by_sequence[batch.frame_seq] = batch;
        const auto max_entries = static_cast<size_t>(std::max(frame_limit * 4, 64));
        while (ground_truth_by_sequence.size() > max_entries) {
            ground_truth_by_sequence.erase(ground_truth_by_sequence.begin());
        }
    }

    std::optional<talos::ipc::GroundTruthBatch> ground_truth_for(uint64_t source_sequence) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto found = ground_truth_by_sequence.find(source_sequence);
        if (found == ground_truth_by_sequence.end()) {
            return std::nullopt;
        }
        return found->second;
    }

    std::filesystem::path root;
    int frame_limit;
    std::ofstream frames;
    std::ofstream candidates;
    std::ofstream fans;
    std::ofstream ground_truth;
    std::map<uint64_t, talos::ipc::GroundTruthBatch> ground_truth_by_sequence;
    std::mutex mutex;
};

template<typename Member>
std::pair<double, double>
avg_p95(const std::vector<Cb5PerfSample>& samples, Member member, bool detected_only = true) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample: samples) {
        if (!detected_only || sample.detected) {
            values.push_back(sample.*member);
        }
    }
    if (values.empty()) {
        return { 0.0, 0.0 };
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    std::sort(values.begin(), values.end());
    const size_t p95_index = static_cast<size_t>(0.95 * static_cast<double>(values.size() - 1));
    return { sum / static_cast<double>(values.size()), values[p95_index] };
}
struct AutoExposureCfg {
    double target_brightness = 0.0;
    double step_gain = 0.0;
    double decay_step = 0.0;
    double tolerance = 0.0;
    double exposure_min = 0.0;
    double exposure_max = 0.0;
    double control_interval_ms = 0.0;

    explicit AutoExposureCfg(const YAML::Node& c):
        target_brightness(c["target_brightness"].as<double>()),
        step_gain(c["step_gain"].as<double>()),
        decay_step(c["decay_step"].as<double>()),
        tolerance(c["tolerance"].as<double>()),
        exposure_min(c["exposure_min"].as<double>()),
        exposure_max(c["exposure_max"].as<double>()),
        control_interval_ms(c["control_interval_ms"].as<double>()) {}
};
bool is_web_running() {
    static std::atomic<bool> cached { true };
    utils::dt_once(
        [&]() {
            const int ret = std::system("pgrep -x wust_vision_web > /dev/null 2>&1");
            cached = (ret == 0);
        },
        std::chrono::duration<double>(1.0)
    );
    bool running = cached.load();
#ifdef USE_RERUN
    running = running || rerun_visual::Recorder::instance().enabled();
#endif
    return running;
}
static constexpr auto RECORD_FOLDER_PATH_ARR = utils::concat(ROOT_DIR, "/record/auto_aim");
static constexpr std::string_view RECORD_FOLDER_PATH(RECORD_FOLDER_PATH_ARR.data());

int main(int argc, char** argv) {
    auto start_tp = Clock::now();
    print_banner();
    auto& signal = utils::SignalGuard::instance();
    logger::init(spdlog::level::trace);

    bool debug = false;
    std::string config_path;
    std::string robot_name;
    auto first_arg = utils::get_arg(1, argc, argv);
    if (first_arg) {
        robot_name = first_arg.value();
        config_path = get_robot_config_path(robot_name).value_or(robot_name);
    } else {
        return 1;
    }
    auto second_arg = utils::get_arg(2, argc, argv);
    if (second_arg) {
        debug = second_arg.value() == "true";
    }
    auto config = YAML::LoadFile(config_path);
    const auto rune_debug_config = config["rune_detector"]["debug"];
    const bool rune_force_full_frame =
        rune_debug_config && rune_debug_config["force_full_frame"]
        && rune_debug_config["force_full_frame"].as<bool>();
    const bool rune_log_diagnostics =
        rune_debug_config && rune_debug_config["log_diagnostics"]
        && rune_debug_config["log_diagnostics"].as<bool>();
    std::unique_ptr<Cb5Capture> cb5_capture;
    if (const char* capture_dir = std::getenv("AW_CB5_CAPTURE_DIR")) {
        int capture_limit = 180;
        if (const char* limit = std::getenv("AW_CB5_CAPTURE_LIMIT")) {
            try {
                capture_limit = std::max(0, std::stoi(limit));
            } catch (const std::exception&) {
                AWAKENING_WARN("invalid AW_CB5_CAPTURE_LIMIT: {}, using 180", limit);
            }
        }
        if (*capture_dir != '\0' && capture_limit > 0) {
            cb5_capture =
                std::make_unique<Cb5Capture>(std::filesystem::path(capture_dir), capture_limit);
            AWAKENING_INFO(
                "CB5 capture enabled: {} frames -> {}",
                capture_limit,
                cb5_capture->root.string()
            );
        }
    }
    Scheduler s;
    std::atomic<Mode> mode = str_to_Mode(config["mode"].as<std::string>());
    std::atomic<EnemyColor> enemy_color =
        enemy_color_from_string(config["enemy_color"].as<std::string>());
    std::atomic<double> bullet_speed = config["bullet_speed"].as<double>();
#ifdef USE_ROS2
    rcl::RclcppNode rcl_node("awakening");
    rcl::TF rcl_tf(rcl_node);
#endif
    bool use_daedalus = false;
#ifdef USE_Daedalus
    std::unique_ptr<talos::ipc::ShmClient> daedalus_shm_client;
    if (config["use_sim"].as<bool>()) {
        auto client = talos::ipc::ShmClient::connect();
        if (!client) {
            AWAKENING_ERROR("Failed to connect to talos::ipc::ShmClient");
            return 1;
        } else {
            use_daedalus = true;
            daedalus_shm_client = std::make_unique<talos::ipc::ShmClient>(std::move(*client));
        }
    }
#endif
    std::unique_ptr<SerialDriver> serial;
    if (!use_daedalus && config["serial"]["enable"].as<bool>()) {
        serial = std::make_unique<SerialDriver>(config["serial"], s);
    }

    auto camera_config = config["camera"];
    std::unique_ptr<Camera> camera;
    utils::SignalGuard::add_callback([&]() {
        if (camera) {
            camera->stop();
        }
    });
    if (!use_daedalus) {
        camera = create_camera(camera_config, s, "daheng");
        camera->init();
        if (!camera->is_running()) {
            return 0;
        }
    }
    std::unique_ptr<VideoSaver> video_saver;
    if (config["record"]["enable"].as<bool>()) {
        video_saver = std::make_unique<VideoSaver>(
            VideoSaver::generate_record_filename(RECORD_FOLDER_PATH.data()),
            VideoSaver::Mode::NonBlocking
        );
    }
    CameraInfo camera_info;
    camera_info.load(camera_config["camera_info"]);
    auto_aim::ArmorDetector armor_detector(config["armor_detector"]);
    auto_aim::ArmorTracker armor_tracker(config["armor_tracker"]);
    auto_aim::AutoAimFsmController auto_aim_fsm_controller(config["auto_aim_fsm"]);
    auto_aim::VeryAimer very_aimer(config["very_aimer"]);
    auto_buff::RuneDetector rune_detector(config["rune_detector"]);
    auto_buff::RuneTracker rune_tracker(config["rune_tracker"]);
    auto_buff::RuneAimer rune_aimer(config["rune_aimer"]);
    utils::OrderedQueue<auto_aim::Armors> armors_queue;
    utils::OrderedQueue<TimedRuneDetection> rune_detections_queue;
    utils::SWMR<auto_aim::ArmorTarget> armor_target;
    utils::SWMR<auto_buff::RuneTarget> rune_target;
    BulletPickUp bullet_pick_up(config["bullet_pick_up"]);
    LogCtx log_ctx;
    Cb5PerfCtx cb5_perf_ctx;
    std::optional<VisionDebugCtx> dbg;
    if (debug) {
        dbg.emplace();
        dbg->camera_info_ = camera_info;
    }
    WheelOdometry wheel_odometry(config["wheel_odometry"], Clock::now());
    // Daedalus pose callbacks and the solver run on different scheduler threads.  The
    // sim adapter below is the only writer; guard the shared WheelOdometry snapshot so
    // the solver and debug logger never observe a torn position/timestamp.
    std::mutex wheel_odometry_mutex;
#ifdef USE_Daedalus
    runtime::DaedalusFramePoseStore daedalus_pose_store;
#endif
    auto tf = SimpleRobotTF::create();
    {
        for (auto [from, to]: std::array {
                 std::pair { SimpleFrame::ODOM, SimpleFrame::GIMBAL_ODOM },
                 std::pair { SimpleFrame::GIMBAL_ODOM, SimpleFrame::GIMBAL },
                 std::pair { SimpleFrame::GIMBAL, SimpleFrame::CAMERA },
                 std::pair { SimpleFrame::GIMBAL, SimpleFrame::SHOOT },
                 std::pair { SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV },
             })
        {
            tf->add_edge(from, to);
        }
        ISO3 cv_in_camera = ISO3::Identity();
        cv_in_camera.linear() = R_CV2PHYSICS;
        tf->push(SimpleFrame::CAMERA, SimpleFrame::CAMERA_CV, Clock::now(), cv_in_camera);
        tf->push(
            SimpleFrame::GIMBAL,
            SimpleFrame::CAMERA,
            Clock::now(),
            utils::load_isometry3(config["tf"]["camera_in_gimbal"])
        );
        tf->push(
            SimpleFrame::GIMBAL,
            SimpleFrame::SHOOT,
            Clock::now(),
            utils::load_isometry3(config["tf"]["shoot_in_gimbal"])
        );
    }
    auto serial_send_to_image_microseconds = config["serial_send_to_image_microseconds"].as<int>();
#ifdef USE_Daedalus
    if (daedalus_shm_client) {
        auto daedalus_imgs = s.register_source<CameraIO>("daedalus_img");

        auto push_pose = [&](talos::ipc::PoseIndex idx, SimpleFrame parent, SimpleFrame child) {
            if (auto pose = daedalus_shm_client->recv_pose(idx)) {
                ISO3 T = ISO3::Identity();
                T.translation() << pose->x, pose->y, pose->z;
                T.linear() =
                    Quaternion { pose->qw, pose->qx, pose->qy, pose->qz }.toRotationMatrix();

                if (static_cast<size_t>(idx) < 4) {
                    daedalus_pose_store.record(
                        static_cast<size_t>(idx),
                        pose->frame_seq,
                        pose->timestamp_ns,
                        T
                    );
                }

                tf->push(parent, child, TimePoint(std::chrono::nanoseconds(pose->timestamp_ns)), T);
                if (idx == talos::ipc::PoseIndex::POSE_ODOM) {
                    // The simulation path has no chassis/wheel observation update, so
                    // WheelOdometry otherwise stays at the origin.  The solver subtracts
                    // this position from target_pos before handing the target to the
                    // Aimer, while shoot_in_gimbal_odom is gimbal-relative.  Keeping the
                    // two frames consistent is required for a correct target bearing.
                    std::lock_guard<std::mutex> lock(wheel_odometry_mutex);
                    wheel_odometry.state.x[wheel_motion_model::idx::X] = pose->x;
                    wheel_odometry.state.x[wheel_motion_model::idx::Y] = pose->y;
                    wheel_odometry.state.x[wheel_motion_model::idx::Z] = pose->z;
                    wheel_odometry.state.x[wheel_motion_model::idx::VX] = 0.0;
                    wheel_odometry.state.x[wheel_motion_model::idx::VY] = 0.0;
                    wheel_odometry.state.x[wheel_motion_model::idx::VZ] = 0.0;
                    wheel_odometry.state.timestamp =
                        TimePoint(std::chrono::nanoseconds(pose->timestamp_ns));
                }
            }
        };

        // Capture the source handle and helper closure by value: both are locals of the
        // conditional block, while the scheduler invokes this callback after that block ends.
        s.add_rate_source<>("daedalus_tick", 300.0, [&, daedalus_imgs, push_pose]() {
            static bool has_camera_info = false;

            if (!has_camera_info) {
                auto info = daedalus_shm_client->camera_info();
                has_camera_info = true;
                camera_info.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
                camera_info.camera_matrix.at<double>(0, 0) = info.fx;
                camera_info.camera_matrix.at<double>(1, 1) = info.fy;
                camera_info.camera_matrix.at<double>(0, 2) = info.cx;
                camera_info.camera_matrix.at<double>(1, 2) = info.cy;
                camera_info.distortion_coefficients = cv::Mat(1, 5, CV_64F);
                std::memcpy(
                    camera_info.distortion_coefficients.ptr<double>(),
                    info.distortion,
                    5 * sizeof(double)
                );
            }

            std::optional<ImageFrame> synchronized_image;
            if (auto frame = daedalus_shm_client->recv_image()) {
                // Talos images are zero-copy views into a 3-slot shared-memory pool. The
                // scheduler/detector is asynchronous, so a later producer frame can overwrite
                // the slot while it is still being processed. Copy once at the boundary.
                synchronized_image = ImageFrame {
                    .src_img = frame->image.clone(),
                    .format = PixelFormat::RGB,
                    .timestamp = TimePoint(std::chrono::nanoseconds(frame->timestamp_ns)),
                    .source_sequence = frame->seq,
                };
            }
            if (auto ground_truth = daedalus_shm_client->recv_ground_truth()) {
                if (cb5_capture) {
                    cb5_capture->record_ground_truth(ground_truth.value());
                }
                if (dbg) {
                    dbg->ground_truth.set(ground_truth.value());
                }
            }

            push_pose(talos::ipc::PoseIndex::POSE_CAMERA, SimpleFrame::GIMBAL, SimpleFrame::CAMERA);
            push_pose(talos::ipc::PoseIndex::POSE_MUZZLE, SimpleFrame::GIMBAL, SimpleFrame::SHOOT);
            push_pose(
                talos::ipc::PoseIndex::POSE_GIMBAL,
                SimpleFrame::GIMBAL_ODOM,
                SimpleFrame::GIMBAL
            );
            // Daedalus publishes POSE_ODOM as (gimbal position in odom, identity rotation).
            // Consume it to (a) satisfy the image/pose synchronization handshake and
            // (b) bind ODOM -> GIMBAL_ODOM with the correct world translation.
            push_pose(
                talos::ipc::PoseIndex::POSE_ODOM,
                SimpleFrame::ODOM,
                SimpleFrame::GIMBAL_ODOM
            );

            // Publish only after every exact-sequence/exact-timestamp pose has reached TF.
            // Keep aggregate diagnostics: per-frame warnings can saturate the log precisely
            // when a synchronization fault needs to remain observable.
            if (synchronized_image) {
                static uint64_t received = 0;
                static uint64_t released = 0;
                static std::array<uint64_t, 4> outcomes {};
                static runtime::DaedalusPoseBundleInspection last_inspection;
                static uint64_t last_sequence = 0;
                static uint64_t last_timestamp_ns = 0;
                static auto last_log = Clock::now();
                const auto timestamp_ns = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        synchronized_image->timestamp.time_since_epoch()
                    )
                        .count()
                );
                const auto sequence = synchronized_image->source_sequence;
                ++received;
                last_sequence = sequence;
                last_timestamp_ns = timestamp_ns;
                last_inspection = daedalus_pose_store.inspect(sequence, timestamp_ns);
                ++outcomes[static_cast<size_t>(last_inspection.failure)];
                if (last_inspection.exact()) {
                    ++released;
                    s.runtime_push_source<CameraIO>(
                        daedalus_imgs,
                        [f = std::move(*synchronized_image)]() mutable {
                            return std::make_tuple(
                                std::optional<CameraIO::second_type>(std::move(f))
                            );
                        }
                    );
                }
                const auto now = Clock::now();
                if (now - last_log >= std::chrono::seconds(1)) {
                    const auto& p = last_inspection.poses;
                    AWAKENING_INFO(
                        "daedalus_sync received/released: {}/{} reject missing/seq/ts: {}/{}/{} "
                        "image={}/{} gimbal={}/{} odom={}/{} muzzle={}/{} camera={}/{}",
                        received,
                        released,
                        outcomes[static_cast<size_t>(
                            runtime::DaedalusPoseBundleFailure::MISSING_POSE
                        )],
                        outcomes[static_cast<size_t>(
                            runtime::DaedalusPoseBundleFailure::SEQUENCE_MISMATCH
                        )],
                        outcomes[static_cast<size_t>(
                            runtime::DaedalusPoseBundleFailure::TIMESTAMP_MISMATCH
                        )],
                        last_sequence,
                        last_timestamp_ns,
                        p[0].frame_sequence,
                        p[0].timestamp_ns,
                        p[1].frame_sequence,
                        p[1].timestamp_ns,
                        p[2].frame_sequence,
                        p[2].timestamp_ns,
                        p[3].frame_sequence,
                        p[3].timestamp_ns
                    );
                    received = 0;
                    released = 0;
                    outcomes.fill(0);
                    last_log = now;
                }
            }
        });
    }
#endif
    if (video_saver) {
        s.register_task<CameraIO>("save_video", [&](CameraIO::second_type&& f) {
            if (!f.src_img.empty()) {
                video_saver->write_frame(f.src_img);
            }
            return std::make_tuple(std::optional<CameraIO::second_type>(std::nullopt));
        });
    }
    s.register_task<CameraIO, CommonFrameIo>("push_common_frame", [&](CameraIO::second_type&& f) {
        static int current_id = 0;
        if (f.src_img.empty()) {
            return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::nullopt));
        }
        log_ctx.camera_count++;
        CommonFrame frame {
            .img_frame = std::move(f),
            .id = current_id++,
            .frame_id = std::to_underlying(SimpleFrame::CAMERA_CV),
        };

        return std::make_tuple(std::optional<CommonFrameIo::second_type>(std::move(frame)));
    });

    if (serial) {
        s.register_task<SerialIO>("receive_serial", [&](SerialIO::second_type&& data) {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            auto now = Clock::now();

            log_ctx.serial_count++;
            if (auto robo_opt = ReceiveRobotData::create(data); robo_opt.has_value()) {
                auto robo = robo_opt.value();
                static uint32_t last_pc = -1, delay = 0, last_bullet_count = 0;
                if (robo.time_stamp_pc != last_pc) {
                    last_pc = robo.time_stamp_pc;
                    delay = (std::chrono::duration_cast<std::chrono::microseconds>(now - start_tp)
                                 .count()
                             - robo.time_stamp_pc
                             - (robo.time_stamp_send_micro - robo.time_stamp_receive_micro))
                        / 2.0;
                }

                auto packet_time =
                    now - std::chrono::microseconds(serial_send_to_image_microseconds);
                ISO3 gimbal_in_gimbal_odom = ISO3::Identity();
                gimbal_in_gimbal_odom.linear() = utils::rpy2matrix(
                    Vec3(robo.roll, robo.pitch, robo.yaw),
                    utils::RPYOrder::ZYX
                );
                tf->push(
                    SimpleFrame::GIMBAL_ODOM,
                    SimpleFrame::GIMBAL,
                    packet_time,
                    gimbal_in_gimbal_odom
                );
                if (robo.shutdown) {
                    std::system("shutdown -h now");
                }
                auto gimbal_odom_in_odom = ISO3::Identity();
                gimbal_odom_in_odom.translation() = Vec3(robo.v_x, robo.v_y, robo.v_z);
                // tf->push(
                //     SimpleFrame::ODOM,
                //     SimpleFrame::GIMBAL_ODOM,
                //     packet_time,
                //     gimbal_odom_in_odom
                // );
                enemy_color = EnemyColor(robo.detect_color);
                bullet_speed = robo.bullet_speed;
                robo.update_log(delay);
                if (robo.bullet_count > last_bullet_count) {
                    bullet_pick_up.push_back(Bullet {
                        .fire_time = Clock::now(),
                        .fire_time_shoot_in_odom =
                            tf->pose_a_in_b(SimpleFrame::SHOOT, SimpleFrame::ODOM, Clock::now()),
                        .speed_in_odom = bullet_speed,
                    });
                }
                last_bullet_count = robo.bullet_count;
            }
        });
    }
    if (camera) {
        s.register_task<CommonFrameIo>("auto_exposure", [&](CommonFrameIo::second_type&& frame) {
            static std::mutex mutex;
            std::lock_guard<std::mutex> lock(mutex);
            static std::optional<AutoExposureCfg> auto_exposure_cfg =
                config["auto_exposure"]["enable"].as<bool>()
                ? std::optional<AutoExposureCfg>(std::in_place, config["auto_exposure"])
                : std::nullopt;
            if (auto_exposure_cfg) { // 平均亮度pid
                auto& cfg = auto_exposure_cfg.value();
                utils::dt_once(
                    [&]() {
                        cv::Mat gray;
                        cv::cvtColor(frame.img_frame.src_img, gray, cv::COLOR_BGR2GRAY);
                        double exposure_time = camera->get_ExposureTime();
                        static double last_exposure_time = 0.0;
                        const double diff = cv::mean(gray)[0] - cfg.target_brightness;
                        if (std::fabs(diff) > cfg.tolerance && exposure_time > 0.0) {
                            exposure_time -= diff * cfg.step_gain;
                        } else {
                            exposure_time -= cfg.decay_step;
                        }
                        exposure_time =
                            std::clamp(exposure_time, cfg.exposure_min, cfg.exposure_max);
                        if (std::abs(exposure_time - last_exposure_time) > 10) {
                            camera->set_ExposureTime(exposure_time);
                            last_exposure_time = exposure_time;
                        }
                    },
                    std::chrono::milliseconds((int)cfg.control_interval_ms)
                );
            }
        });
    }

    s.register_task<CommonFrameIo, DetIo>("detector", [&](CommonFrameIo::second_type&& frame) {
        Detection r;
        cv::Rect net_focus =
            cv::Rect(0, 0, frame.img_frame.src_img.cols, frame.img_frame.src_img.rows);
        if (mode == Mode::AutoAim) {
            static std::unique_ptr<std::counting_semaphore<>> detector_sem;
            if (!detector_sem) {
                detector_sem = std::make_unique<std::counting_semaphore<>>(
                    config["armor_detector"]["max_infer_num"].as<int>()
                );
            }
            std::optional<cv::Rect> detect_light = std::nullopt;
            auto target = armor_target.read();
            if (target.need_focus()) {
                auto camera_cv_in_old = tf->pose_a_in_b(
                    SimpleFrame(frame.frame_id),
                    SimpleFrame(target.get_target_state().frame_id),
                    frame.img_frame.timestamp
                );
                net_focus = target.get_net_focus_roi(
                    frame.img_frame.timestamp,
                    camera_cv_in_old,
                    camera_info,
                    frame.img_frame.src_img.size(),
                    armor_detector.get_net_wh_ratio()
                );
                if (target.need_detect_lights()) {
                    detect_light = target.expanded( // 送给传统越小越好
                        frame.img_frame.timestamp,
                        camera_cv_in_old,
                        camera_info,
                        frame.img_frame.src_img.size()
                    );
                    detect_light = utils::expand_and_clip_rect(
                        detect_light.value(),
                        1.6,
                        frame.img_frame.src_img.size()
                    );
                }
            }

            auto_aim::Armors armors { .timestamp = frame.img_frame.timestamp,
                                      .id = frame.id,
                                      .frame_id = frame.frame_id };
            {
                bool got = detector_sem->try_acquire();
                utils::SemaphoreGuard guard(*detector_sem, got); //并发控制
                if (got) {
                    auto start = Clock::now();
                    r.detect_start = start;
                    auto [ls, as] = armor_detector.detect(frame, net_focus, detect_light);
                    armors.armors = as;
                    armors.lights = ls;
                    log_ctx.detect_count++;
                }
            }
            armors_queue.enqueue(armors);
            auto batch_armors = armors_queue.dequeue_batch(); // 根据id有序输出
            r.armors = batch_armors;
        } else {
            static std::unique_ptr<std::counting_semaphore<>> detector_sem;
            if (!detector_sem) {
                detector_sem = std::make_unique<std::counting_semaphore<>>(
                    config["rune_detector"]["max_infer_num"].as<int>()
                );
            }
            auto target = rune_target.read();
            if (!rune_force_full_frame && target.need_focus()) {
                auto camera_cv_in_old = tf->pose_a_in_b(
                    SimpleFrame(frame.frame_id),
                    SimpleFrame(target.get_target_state().frame_id),
                    frame.img_frame.timestamp
                );
                net_focus = target.get_net_focus_roi(
                    frame.img_frame.timestamp,
                    camera_cv_in_old,
                    camera_info,
                    frame.img_frame.src_img.size(),
                    rune_detector.get_net_wh_ratio()
                );
            }
            TimedRuneDetection timed_rune;
            timed_rune.detector_start = Clock::now();
            timed_rune.source_sequence = frame.img_frame.source_sequence;
            if (cb5_capture && cb5_capture->includes(frame.id)) {
                cv::Mat saved;
                if (frame.img_frame.format == PixelFormat::RGB) {
                    cv::cvtColor(frame.img_frame.src_img, saved, cv::COLOR_RGB2BGR);
                } else {
                    saved = frame.img_frame.src_img;
                }
                std::ostringstream name;
                name << "frame_" << std::setw(6) << std::setfill('0') << frame.id << "_seq_"
                     << frame.img_frame.source_sequence << "_ts_"
                     << std::chrono::duration_cast<std::chrono::nanoseconds>(
                            frame.img_frame.timestamp.time_since_epoch()
                        )
                            .count()
                     << ".png";
                if (!cv::imwrite((cb5_capture->root / "images" / name.str()).string(), saved)) {
                    AWAKENING_WARN("failed to save CB5 capture image {}", name.str());
                }
            }
            {
                bool got = detector_sem->try_acquire();
                utils::SemaphoreGuard guard(*detector_sem, got); //并发控制
                if (got) {
                    timed_rune.detected = true;
                    r.detect_start = timed_rune.detector_start;
                    timed_rune.detection = rune_detector.detect(frame, net_focus, enemy_color);
                    log_ctx.detect_count++;
                }
            }
            timed_rune.detector_end = Clock::now();
            timed_rune.focus = net_focus;
            timed_rune.full_frame = net_focus
                == cv::Rect(
                    0,
                    0,
                    frame.img_frame.src_img.cols,
                    frame.img_frame.src_img.rows
                );
            timed_rune.detection.frame_id = frame.frame_id;
            timed_rune.detection.id = frame.id;
            timed_rune.detection.timestamp = frame.img_frame.timestamp;
            timed_rune.id = frame.id;
            rune_detections_queue.enqueue(std::move(timed_rune));
            auto batch_runes = rune_detections_queue.dequeue_batch();
            r.rune = batch_runes;
        }

        if (dbg && is_web_running()) {
            dbg->expanded.set(net_focus);
            frame.img_frame.detector_sequence = static_cast<uint64_t>(frame.id);
            frame.img_frame.debug_cache_timestamp = Clock::now();
            // Clear overlays before publishing a new raw frame. The tracker may replace this
            // empty result later in the same pipeline pass, but an old target is never drawn on
            // a newer no-detection image.
            if (mode == Mode::AutoAim) {
                dbg->armors.set(auto_aim::Armors {
                    .timestamp = frame.img_frame.timestamp,
                    .id = frame.id,
                    .frame_id = frame.frame_id,
                });
            } else {
                auto_buff::RuneDetection empty_rune;
                empty_rune.timestamp = frame.img_frame.timestamp;
                empty_rune.id = frame.id;
                empty_rune.frame_id = frame.frame_id;
                dbg->rune.set(std::move(empty_rune));
            }
            dbg->img_frame.set(std::move(frame.img_frame));
        }

        return std::make_tuple(std::optional<DetIo::second_type>(std::move(r)));
    });

    s.register_task<DetIo>("tracker", [&](DetIo::second_type&& io) {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (io.armors) {
            for (const auto& armors_raw: io.armors.value()) {
                auto armors = armors_raw;
                auto is_ally = [&](const auto& obj) {
                    return (enemy_color == EnemyColor::BLUE
                            && obj.color == auto_aim::ArmorColor::RED)
                        || (enemy_color == EnemyColor::RED
                            && obj.color == auto_aim::ArmorColor::BLUE);
                };
                armors.armors.erase(
                    std::remove_if(armors.armors.begin(), armors.armors.end(), is_ally),
                    armors.armors.end()
                );
                armors.lights.erase(
                    std::remove_if(armors.lights.begin(), armors.lights.end(), is_ally),
                    armors.lights.end()
                );
                auto camera_cv_in_odom = tf->pose_a_in_b(
                    SimpleFrame(armors.frame_id),
                    SimpleFrame::ODOM,
                    armors.timestamp
                ); //转到odom
                armors.frame_id = std::to_underlying(SimpleFrame::ODOM);
                auto __armor_target =
                    armor_tracker.track(armors, camera_info, camera_cv_in_odom, armors.frame_id);
                auto_aim_fsm_controller.update(
                    __armor_target.get_target_state().vyaw(),
                    __armor_target.jumped,
                    __armor_target.get_target_state().timestamp
                );
                armor_target.write(__armor_target);
                if (io.detect_start) {
                    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          Clock::now() - io.detect_start.value()
                    )
                                          .count();
                    log_ctx.latency_ms_total += latency_ms;
                }

                log_ctx.found_count += armor_tracker.get_count();
                armor_tracker.reset_count();
                if (dbg) {
                    dbg->armors.set(armors);
                }

                log_ctx.track_count++;
            }
        } else if (io.rune) {
            for (auto& timed_rune: io.rune.value()) {
                const auto tracker_start = Clock::now();
                auto& rune = timed_rune.detection;
                auto camera_cv_in_odom = tf->pose_a_in_b(
                    SimpleFrame(rune.frame_id),
                    SimpleFrame::ODOM,
                    rune.timestamp
                ); //转到odom
                rune.frame_id = std::to_underlying(SimpleFrame::ODOM);

                auto __rune_target =
                    rune_tracker.track(rune, camera_info, camera_cv_in_odom, rune.frame_id);
                const auto tracker_diagnostics = rune_tracker.diagnostics();

                rune_target.write(__rune_target);
                if (cb5_capture && cb5_capture->includes(rune.id)) {
                    const auto ground_truth =
                        cb5_capture->ground_truth_for(timed_rune.source_sequence);
                    runtime::DaedalusFramePoseSnapshots pose_snapshots {};
#ifdef USE_Daedalus
                    pose_snapshots =
                        daedalus_pose_store.snapshot(timed_rune.source_sequence);
#endif
                    const auto image_timestamp_ns =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            rune.timestamp.time_since_epoch()
                        )
                            .count();
                    bool pose_sync_exact = timed_rune.source_sequence != 0;
                    for (const auto& pose: pose_snapshots) {
                        pose_sync_exact = pose_sync_exact && pose.valid
                            && pose.frame_sequence == timed_rune.source_sequence
                            && pose.timestamp_ns == static_cast<uint64_t>(image_timestamp_ns);
                    }
                    const Quaternion camera_odom_q(camera_cv_in_odom.rotation());
                    const auto tracker_position = __rune_target.get_target_state().pos();
                    std::lock_guard<std::mutex> capture_lock(cb5_capture->mutex);
                    const auto& infer = rune.infer_diagnostics;
                    const auto stamp = [&](size_t idx) -> const runtime::DaedalusPoseSnapshot& {
                        return pose_snapshots[idx];
                    };
                    cb5_capture->frames
                        << rune.id << ',' << timed_rune.source_sequence << ','
                        << image_timestamp_ns << ',' << timed_rune.detected << ','
                        << infer.raw_anchors << ',' << infer.class_candidates << ','
                        << infer.rejected_by_configured_min << ',' << infer.rejected_invalid_r
                        << ',' << infer.rejected_invalid_cardinal << ','
                        << infer.rejected_by_aw_strict_rule << ',' << infer.aw_before_nms << ','
                        << infer.aw_after_nms << ',' << infer.original_before_nms << ','
                        << infer.original_after_nms << ',' << rune.fan_blades.size() << ','
                        << pose_sync_exact << ',' << stamp(3).frame_sequence << ','
                        << stamp(3).timestamp_ns << ',' << stamp(0).frame_sequence << ','
                        << stamp(0).timestamp_ns << ',' << stamp(1).frame_sequence << ','
                        << stamp(1).timestamp_ns << ',' << stamp(2).frame_sequence << ','
                        << stamp(2).timestamp_ns << ','
                        << camera_cv_in_odom.translation().x() << ','
                        << camera_cv_in_odom.translation().y() << ','
                        << camera_cv_in_odom.translation().z() << ',' << camera_odom_q.w() << ','
                        << camera_odom_q.x() << ',' << camera_odom_q.y() << ','
                        << camera_odom_q.z() << ',' << camera_info.camera_fx() << ','
                        << camera_info.camera_fy() << ',' << camera_info.camera_cx() << ','
                        << camera_info.camera_cy();
                    for (int coefficient = 0; coefficient < 5; ++coefficient) {
                        cb5_capture->frames << ','
                                            << camera_info.distortion_coefficients.at<double>(
                                                   0,
                                                   coefficient
                                               );
                    }
                    cb5_capture->frames << ',' << talos::ipc::IMAGE_WIDTH << ','
                                        << talos::ipc::IMAGE_HEIGHT << ','
                        << tracker_position.x() << ','
                        << tracker_position.y() << ',' << tracker_position.z() << ','
                        << static_cast<int>(__rune_target.track_state.tracker_state) << ','
                        << tracker_diagnostics.valid_matches << '\n';
                    const auto write_ground_truth_row = [&](int rune_index,
                                                            const talos::ipc::GroundTruthRune* gt) {
                        const bool batch_exact = ground_truth
                            && ground_truth->frame_seq == timed_rune.source_sequence
                            && ground_truth->timestamp_ns
                                == static_cast<uint64_t>(image_timestamp_ns);
                        const bool exact = batch_exact && gt
                            && gt->frame_seq == timed_rune.source_sequence
                            && gt->timestamp_ns == static_cast<uint64_t>(image_timestamp_ns);
                        cb5_capture->ground_truth
                            << rune.id << ',' << timed_rune.source_sequence << ','
                            << image_timestamp_ns << ',' << static_cast<bool>(ground_truth) << ','
                            << exact << ',' << (ground_truth ? ground_truth->frame_seq : 0) << ','
                            << (ground_truth ? ground_truth->timestamp_ns : 0) << ','
                            << rune_index;
                        if (!gt) {
                            for (int column = 0; column < 23; ++column) {
                                cb5_capture->ground_truth << ',';
                            }
                            cb5_capture->ground_truth << '\n';
                            return;
                        }
                        cb5_capture->ground_truth
                            << ',' << gt->frame_seq << ',' << gt->timestamp_ns << ','
                            << static_cast<int>(gt->team) << ','
                            << static_cast<int>(gt->rune_mode) << ','
                            << static_cast<int>(gt->mechanism_state) << ','
                            << gt->r_center_odom[0] << ',' << gt->r_center_odom[1] << ','
                            << gt->r_center_odom[2] << ',' << gt->radius << ','
                            << gt->current_angle << ',' << gt->v_roll << ',' << gt->direction
                            << ',' << gt->sin_amplitude << ',' << gt->sin_omega << ','
                            << gt->sin_phase << ',' << gt->sin_offset << ','
                            << gt->relative_time << ',' << gt->blade_id;
                        for (const auto activation: gt->target_activations) {
                            cb5_capture->ground_truth << ',' << static_cast<int>(activation);
                        }
                        cb5_capture->ground_truth << '\n';
                    };
                    if (!ground_truth || ground_truth->rune_count == 0) {
                        write_ground_truth_row(-1, nullptr);
                    } else {
                        const auto count = std::min<size_t>(
                            ground_truth->rune_count,
                            talos::ipc::GROUND_TRUTH_MAX_RUNES
                        );
                        for (size_t rune_index = 0; rune_index < count; ++rune_index) {
                            write_ground_truth_row(
                                static_cast<int>(rune_index),
                                &ground_truth->runes[rune_index]
                            );
                        }
                    }
                    for (const auto& candidate: infer.candidates) {
                        cb5_capture->candidates
                            << rune.id << ',' << timed_rune.source_sequence << ','
                            << candidate.anchor << ',' << candidate.class_id << ','
                            << candidate.class_confidence << ',' << candidate.valid_keypoints << ','
                            << candidate.rejected_by_configured_min << ','
                            << candidate.rejected_invalid_r << ','
                            << candidate.rejected_invalid_cardinal << ','
                            << candidate.accepted_by_aw << ','
                            << candidate.accepted_by_original_policy;
                        for (size_t k = 0; k < candidate.raw_points.size(); ++k) {
                            cb5_capture->candidates
                                << ',' << candidate.raw_points[k].x << ','
                                << candidate.raw_points[k].y << ','
                                << candidate.keypoint_confidences[k];
                        }
                        cb5_capture->candidates << '\n';
                    }
                    for (size_t fan_index = 0; fan_index < rune.fan_blades.size(); ++fan_index) {
                        const auto& fan = rune.fan_blades[fan_index];
                        cb5_capture->fans
                            << rune.id << ',' << timed_rune.source_sequence << ',' << fan_index
                            << ',' << static_cast<int>(fan.activation) << ',' << fan.confidence
                            << ',' << fan.pose_valid << ',' << fan.reprojection_error_px << ','
                            << fan.pose_camera_cv.translation().x() << ','
                            << fan.pose_camera_cv.translation().y() << ','
                            << fan.pose_camera_cv.translation().z() << ','
                            << fan.pose.translation().x() << ',' << fan.pose.translation().y()
                            << ',' << fan.pose.translation().z() << ','
                            << fan.tracker_predicted_position.x() << ','
                            << fan.tracker_predicted_position.y() << ','
                            << fan.tracker_predicted_position.z() << ',' << fan.match_best_id << ','
                            << fan.match_id << ',' << fan.match_best_d2;
                        for (double residual: fan.match_residual_ypd_rotation) {
                            cb5_capture->fans << ',' << residual;
                        }
                        for (size_t k = 0; k < fan.raw_points.size(); ++k) {
                            cb5_capture->fans << ',' << fan.raw_points[k].x << ','
                                              << fan.raw_points[k].y << ','
                                              << fan.raw_confidences[k];
                        }
                        cb5_capture->fans << '\n';
                    }
                    cb5_capture->frames.flush();
                    cb5_capture->candidates.flush();
                    cb5_capture->fans.flush();
                    cb5_capture->ground_truth.flush();
                }
                const auto tracker_end = Clock::now();
                const auto& timing = rune.timing;
                cb5_perf_ctx.add(Cb5PerfSample {
                    .detected = timed_rune.detected,
                    .preprocess_ms = timing.preprocess_ms,
                    .inference_ms = timing.inference_ms,
                    .output_copy_ms = timing.output_copy_ms,
                    .postprocess_ms = timing.postprocess_ms,
                    .traditional_cv_ms = timing.traditional_cv_ms,
                    .detector_ms = timing.detector_ms,
                    .queue_wait_ms = std::chrono::duration<double, std::milli>(
                                         tracker_start - timed_rune.detector_end
                    )
                                         .count(),
                    .tracker_ms = std::chrono::duration<double, std::milli>(
                                      tracker_end - tracker_start
                    )
                                      .count(),
                    .total_ms = timed_rune.detected
                        ? std::chrono::duration<double, std::milli>(
                              tracker_end - timed_rune.detector_start
                          )
                              .count()
                        : 0.0,
                    .focus = timed_rune.focus,
                    .full_frame = timed_rune.full_frame,
                    .fan_blades = tracker_diagnostics.fan_blades,
                    .pose_valid_fan_blades = tracker_diagnostics.pose_valid_fan_blades,
                    .valid_matches = tracker_diagnostics.valid_matches,
                    .initialized = tracker_diagnostics.initialized,
                    .tracker_state = __rune_target.track_state.tracker_state,
                    .clock_wise_count = __rune_target.voter.clock_wise_count,
                    .double_detect_count = __rune_target.voter.double_detect_count,
                    .voter_state = __rune_target.voter.state,
                    .voter_mode = __rune_target.voter.mode,
                });
                if (io.detect_start) {
                    auto latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                          Clock::now() - io.detect_start.value()
                    )
                                          .count();
                    log_ctx.latency_ms_total += latency_ms;
                }
                log_ctx.found_count += rune_tracker.get_count();
                rune_tracker.reset_count();

                if (dbg) {
                    dbg->rune.set(rune);
                }
                log_ctx.track_count++;
            }
        }
    });
    s.add_rate_source<>("solver", 1000.0, [&]() {
        log_ctx.solve_count++;
        auto a_target = armor_target.read(); //需要转为相对gimbal_odom
        int old_this_id = a_target.this_id;
        wheel_motion_model::State gimbal_odom_state_in_odom;
        {
            std::lock_guard<std::mutex> lock(wheel_odometry_mutex);
            gimbal_odom_state_in_odom = wheel_odometry.state;
        }
        gimbal_odom_state_in_odom.predict(Clock::now());
        a_target.set_target_state([&](auto& s) {
            namespace idx = auto_aim::armor_point_motion_model::idx;
            s.frame_id = std::to_underlying(SimpleFrame::GIMBAL_ODOM);
            const auto pos = gimbal_odom_state_in_odom.pos();
            const auto vel = gimbal_odom_state_in_odom.vel();
            s.x[idx::CX] -= pos.x();
            s.x[idx::CY] -= pos.y();
            s.x[idx::CZ] -= pos.z();
            s.x[idx::VCX] -= vel.x();
            s.x[idx::VCY] -= vel.y();
            s.x[idx::VCZ] -= vel.z();
            // s.x[idx::VCX] =0;
            // s.x[idx::VCY] =0;
            // s.x[idx::VCZ] =0;
            // s.x[idx::VYAW] =12.0;
        });
        a_target.this_id = old_this_id;
        auto r_target = rune_target.read();
        old_this_id = r_target.this_id;
        r_target.set_target_state([&](auto& s) {
            namespace idx = auto_buff::motion_model::idx;
            s.frame_id = std::to_underlying(SimpleFrame::GIMBAL_ODOM);
            const auto pos = gimbal_odom_state_in_odom.pos();
            const auto vel = gimbal_odom_state_in_odom.vel();
            s.x[idx::CX] -= pos.x();
            s.x[idx::CY] -= pos.y();
            s.x[idx::CZ] -= pos.z();
        });
        r_target.this_id = old_this_id;
        GimbalCmd cmd {
            .appear = false,
        };
        auto shoot_in_gimbal_odom =
            tf->pose_a_in_b(SimpleFrame::SHOOT, SimpleFrame::GIMBAL_ODOM, Clock::now());
        auto gimbal_in_gimbal_odom =
            tf->pose_a_in_b(SimpleFrame::GIMBAL, SimpleFrame::GIMBAL_ODOM, Clock::now());
        static ShootControlSend shoot_send;
        if (a_target.check()) {
            cmd = very_aimer.very_aim(
                a_target,
                bullet_speed,
                auto_aim_fsm_controller.get_state(),
                shoot_in_gimbal_odom,
                gimbal_in_gimbal_odom
            );

        } else if (r_target.check()) {
            cmd =
                rune_aimer.aim(r_target, bullet_speed, shoot_in_gimbal_odom, gimbal_in_gimbal_odom);
            bool can_fire = rune_aimer.tick_fire(r_target, cmd);
            if (can_fire) {
                AWAKENING_INFO("you can fire!");
                cmd.fire_advice = true;
                shoot_send.shoot_count++;
            } else {
                cmd.fire_advice = false;
            }
        }

        if (serial) {
            SendRobotCmdData send;
            send.cmd_ID = SendRobotCmdData::ID;
            send.time_stamp =
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start_tp)
                    .count();
            send.appear = cmd.appear, send.detect_color = std::to_underlying(enemy_color.load());
            send.yaw = cmd.yaw, send.pitch = cmd.pitch, send.v_yaw = cmd.v_yaw;
            send.target_yaw = cmd.target_yaw, send.target_pitch = cmd.target_pitch;
            send.v_pitch = cmd.v_pitch, send.a_yaw = cmd.a_yaw, send.a_pitch = cmd.a_pitch;
            send.enable_yaw_diff = cmd.enable_yaw_diff;
            send.enable_pitch_diff = cmd.enable_pitch_diff;
            send.track_id = static_cast<uint8_t>(a_target.target_number);
            Web::write_log("send_cmd", [&](auto& j) {
                j["appear"] = Web::val((int)send.appear);
            });
            if (utils::is_finite(send)) {
                serial->write(std::move(utils::to_vector(send)));
            }
            // if (utils::is_finite(shoot_send)) {
            //     serial->write(std::move(utils::to_vector(shoot_send)));
            // }
        }
#ifdef USE_Daedalus
        if (daedalus_shm_client) {
            // Simulator safety: never forward a firing command to Daedalus.
            constexpr bool send_fire = false;
            // Team mpc can emit NaN control states in this cold-start sim. Fall back to
            // its finite target trajectory at the adapter boundary; do not alter the
            // team Aimer implementation or its mpc configuration.
            const double out_yaw = std::isfinite(cmd.yaw) ? cmd.yaw : cmd.target_yaw;
            const double out_pitch = std::isfinite(cmd.pitch) ? cmd.pitch : -cmd.target_pitch;
            // very_aimer works in radians; the Talos GimbalCmd interface is in degrees.
            // The Aimer already publishes the gimbal yaw in the GIMBAL_ODOM frame;
            // adding an empirical +90 here rotates the target away from the armor.
            const double sent_yaw = angles::normalize_degrees(angles::to_degrees(out_yaw));
            daedalus_shm_client->send_gimbal_cmd(
                sent_yaw,
                -angles::to_degrees(out_pitch),
                cmd.appear ? 1.0 : -1.0,
                send_fire
            );
        }
#endif
        auto old_in_camera_cv = tf->pose_a_in_b(
            SimpleFrame(cmd.aim_point.frame_id),
            SimpleFrame::CAMERA_CV,
            cmd.timestamp
        );
        cmd.aim_point.transform(old_in_camera_cv, std::to_underlying(SimpleFrame::CAMERA_CV));
        if (dbg && is_web_running()) {
            dbg->gimbal_cmd.set(cmd);
        }
    });
    s.add_rate_source<>("logger", 1.0, [&]() {
        double avg_latency_ms = log_ctx.latency_ms_total / log_ctx.detect_count;
        AWAKENING_INFO(
            "detect: {} track: {} found: {} solve: {} serial: {} camera: {} avg_latency: {:.3} ms",
            log_ctx.detect_count,
            log_ctx.track_count,
            log_ctx.found_count,
            log_ctx.solve_count,
            log_ctx.serial_count,
            log_ctx.camera_count,
            avg_latency_ms
        );
        if (dbg) {
            dbg->avg_latency_ms.set(avg_latency_ms);
        }
        if (mode == Mode::AutoBuff) {
            const auto samples = cb5_perf_ctx.take();
            if (!samples.empty()) {
                const auto detected = static_cast<int>(std::count_if(
                    samples.begin(),
                    samples.end(),
                    [](const auto& sample) { return sample.detected; }
                ));
                const auto pre = avg_p95(samples, &Cb5PerfSample::preprocess_ms);
                const auto infer = avg_p95(samples, &Cb5PerfSample::inference_ms);
                const auto copy = avg_p95(samples, &Cb5PerfSample::output_copy_ms);
                const auto post = avg_p95(samples, &Cb5PerfSample::postprocess_ms);
                const auto cv = avg_p95(samples, &Cb5PerfSample::traditional_cv_ms);
                const auto detector = avg_p95(samples, &Cb5PerfSample::detector_ms);
                const auto queue = avg_p95(samples, &Cb5PerfSample::queue_wait_ms);
                const auto tracker = avg_p95(samples, &Cb5PerfSample::tracker_ms, false);
                const auto total = avg_p95(samples, &Cb5PerfSample::total_ms);
                AWAKENING_INFO(
                    "cb5_perf tracked: {} detected: {} skipped: {} "
                    "pre avg/p95: {:.3f}/{:.3f} infer: {:.3f}/{:.3f} "
                    "copy: {:.3f}/{:.3f} post: {:.3f}/{:.3f}",
                    samples.size(),
                    detected,
                    static_cast<int>(samples.size()) - detected,
                    pre.first,
                    pre.second,
                    infer.first,
                    infer.second,
                    copy.first,
                    copy.second,
                    post.first,
                    post.second
                );
                AWAKENING_INFO(
                    "cb5_perf cv avg/p95: {:.3f}/{:.3f} detector: {:.3f}/{:.3f} "
                    "queue: {:.3f}/{:.3f} tracker_all: {:.3f}/{:.3f} "
                    "total: {:.3f}/{:.3f}",
                    cv.first,
                    cv.second,
                    detector.first,
                    detector.second,
                    queue.first,
                    queue.second,
                    tracker.first,
                    tracker.second,
                    total.first,
                    total.second
                );
                if (rune_log_diagnostics) {
                    const auto fan_blades = std::accumulate(
                        samples.begin(),
                        samples.end(),
                        0,
                        [](int sum, const auto& sample) { return sum + sample.fan_blades; }
                    );
                    const auto pose_valid = std::accumulate(
                        samples.begin(),
                        samples.end(),
                        0,
                        [](int sum, const auto& sample) {
                            return sum + sample.pose_valid_fan_blades;
                        }
                    );
                    const auto valid_matches = std::accumulate(
                        samples.begin(),
                        samples.end(),
                        0,
                        [](int sum, const auto& sample) { return sum + sample.valid_matches; }
                    );
                    const auto init_count = static_cast<int>(std::count_if(
                        samples.begin(),
                        samples.end(),
                        [](const auto& sample) { return sample.initialized; }
                    ));
                    const auto dynamic_roi_count = static_cast<int>(std::count_if(
                        samples.begin(),
                        samples.end(),
                        [](const auto& sample) { return !sample.full_frame; }
                    ));
                    const auto& last = samples.back();
                    AWAKENING_INFO(
                        "cb5_diag roi_mode: {} dynamic/full: {}/{} last_roi: [{},{},{},{}] "
                        "fan_blades: {} pose_valid: {} valid_matches: {} init: {} "
                        "tracker: {} clock_wise_count: {} double_detect_count: {} "
                        "voter_state/mode: {}/{}",
                        rune_force_full_frame ? "forced_full" : "dynamic",
                        dynamic_roi_count,
                        static_cast<int>(samples.size()) - dynamic_roi_count,
                        last.focus.x,
                        last.focus.y,
                        last.focus.width,
                        last.focus.height,
                        fan_blades,
                        pose_valid,
                        valid_matches,
                        init_count,
                        auto_buff::RuneTarget::TrackState::string_by_state(last.tracker_state),
                        last.clock_wise_count,
                        last.double_detect_count,
                        last.voter_state,
                        last.voter_mode
                    );
                }
            }
        }
        log_ctx.reset();
    });
    if (dbg) {
        s.add_rate_source<>("debug", 45.0, [&]() {
            auto __armor_target = armor_target.read();
            if (is_web_running()) {
                static Web web;
                dbg->type =
                    (mode == Mode::AutoAim) ? VisionDebugCtx::AUTO_AIM : VisionDebugCtx::AUTO_BUFF;
                auto __rune_target = rune_target.read();
                __armor_target.write_log();
                __rune_target.write_log();
                {
                    std::lock_guard<std::mutex> lock(wheel_odometry_mutex);
                    wheel_odometry.write_log();
                }
                auto img_now = dbg->img_frame.get().timestamp;
                dbg->armor_target.set(__armor_target);
                dbg->rune_target.set(__rune_target);
                dbg->auto_aim_fsm_state.set(auto_aim_fsm_controller.get_state());
                auto gimbal_in_gimbal_odom =
                    tf->pose_a_in_b(SimpleFrame::GIMBAL, SimpleFrame::GIMBAL_ODOM, Clock::now());
                auto rpy =
                    utils::matrix2rpy<double>(gimbal_in_gimbal_odom.linear(), utils::RPYOrder::ZYX);
                auto gimbal_yaw_pitch =
                    std::make_pair(rpy[2], -rpy[1]);
                dbg->gimbal_yaw_pitch.set(gimbal_yaw_pitch);
                bullet_pick_up.update(
                    Clock::now(),
                    dbg->gimbal_cmd.get().appear ? dbg->gimbal_cmd.get().fly_time : 0.4
                );
                auto bullet_poss =
                    bullet_pick_up.get_bullet_positions(img_now, very_aimer.get_yaw_pitch_offset());
                auto odom_in_camera_cv =
                    tf->pose_a_in_b(SimpleFrame::ODOM, SimpleFrame::CAMERA_CV, img_now);
                for (auto& pos: bullet_poss) {
                    pos = odom_in_camera_cv * pos;
                }
                dbg->odom_in_camera_cv.set(odom_in_camera_cv);
                dbg->bullet_positions.set(bullet_poss);
                web.write_debug_data(dbg.value());
                auto img = dbg->img_frame.get();
                auto debug_img = img.src_img.clone();
                if (img.format == PixelFormat::RGB) {
                    cv::cvtColor(debug_img, debug_img, cv::COLOR_RGB2BGR);
                }
                if (!debug_img.empty()) {
                    web.draw(debug_img, dbg.value());
                    const auto web_sequence = web.write_shm(debug_img, img);
                    static auto last_frame_diag = Clock::now();
                    const auto now = Clock::now();
                    if (now - last_frame_diag >= std::chrono::seconds(1)) {
                        const auto age_ms =
                            std::chrono::duration<double, std::milli>(
                                now - img.debug_cache_timestamp
                            )
                                .count();
                        AWAKENING_INFO(
                            "visual_frame capture={} detector={} web={} cache_age_ms={:.1f} stale={}",
                            img.source_sequence,
                            img.detector_sequence,
                            web_sequence,
                            age_ms,
                            age_ms > 500.0
                        );
                        last_frame_diag = now;
                    }
                }
            }

#ifdef USE_ROS2
            auto armors = dbg->armors.get();
            rcl::pub_armor_target_marker(
                rcl_node,
                SimpleFrame_to_str(__armor_target.get_target_state().frame_id),
                __armor_target
            );
#endif
        });
#ifdef USE_ROS2
        if (debug) {
            s.add_rate_source<>("tf_pub", 100.0, [&]() {
                rcl_tf.pub_robot_tf(*tf, [](SimpleFrame frame) {
                    return SimpleFrame_to_str(frame);
                });
            });
        }

#endif
#ifdef USE_RERUN
        if (debug) {
            s.add_rate_source<>("rerun_tf", 15.0, [&]() {
                rerun_visual::log_robot_tf(*tf, [](SimpleFrame frame) {
                    return SimpleFrame_to_str(frame);
                });
            });
        }
#endif
    }

    if (camera) {
        camera->start<CameraTag>("hik");
    }
    if (serial) {
        serial->start<SerialTag>("serial");
    }
    s.build();
    s.run();

#ifdef USE_ROS2
    std::thread([&]() { rcl_node.spin(); }).detach();
#endif
    utils::SignalGuard::spin(std::chrono::milliseconds(1000));
    s.stop();
#ifdef USE_ROS2
    rcl_node.shutdown();
#endif
    for (int i = 0; i < 10; ++i) {
        AWAKENING_CRITICAL("改了东西记得同步其他有关的exe的src");
    }
    return 0;
}
