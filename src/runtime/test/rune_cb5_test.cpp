#include "tasks/auto_buff/rune_detect/rune_detector.hpp"
#include "tasks/auto_buff/rune_detect/rune_infer.hpp"
#include "tasks/auto_buff/rune_track/rune_target.hpp"
#include "tasks/auto_buff/type.hpp"
#include "tasks/auto_aim/armor_detect/armor_detector.hpp"
#include "tasks/base/common.hpp"
#include "utils/common/image.hpp"
#include "utils/logger.hpp"
#include "utils/net_detector/openvino/net_detector_openvino.hpp"
#ifdef USE_TRT
    #include "utils/net_detector/tensorrt/net_detector_tensorrt.hpp"
#endif
#include "utils/utils.hpp"
#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace awakening;
using namespace awakening::auto_buff;

const char* kpt_name(int i) {
    static constexpr const char* names[] = { "TOP", "LEFT", "BOTTOM", "RIGHT", "CENTER", "R" };
    return i >= 0 && i < 6 ? names[i] : "?";
}

void draw_point(cv::Mat& img, const cv::Point2f& p, const cv::Scalar& color, const std::string& text) {
    cv::circle(img, p, 5, color, cv::FILLED);
    cv::putText(
        img,
        text,
        cv::Point(static_cast<int>(p.x) + 6, static_cast<int>(p.y) - 6),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv::LINE_AA
    );
}

void process_image(
    RuneDetector& detector,
    const CameraInfo& camera_info,
    const std::string& image_path,
    const std::string& output_dir,
    EnemyColor enemy_color
) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "cannot read image: " << image_path << "\n";
        return;
    }
    CommonFrame frame;
    frame.img_frame.src_img = img;
    frame.img_frame.format = PixelFormat::BGR;
    frame.img_frame.timestamp = Clock::now();
    frame.id = 0;
    frame.frame_id = 0;

    auto detection = detector.detect(frame, cv::Rect(0, 0, img.cols, img.rows), enemy_color);
    std::cout << "\n=== " << image_path << " ===\n";
    std::cout << "fan_blades=" << detection.fan_blades.size()
              << " fan_targets=" << detection.fan_targets.size()
              << " rune_rs=" << detection.rune_rs.size() << "\n";

    cv::Mat vis = img.clone();
    int idx = 0;
    for (auto& fan: detection.fan_blades) {
        std::cout << "  [" << idx++ << "] activation=" << rune_activation_name(fan.activation)
                  << " conf=" << fan.confidence
                  << " geometry=" << (fan.geometry == RuneBladeGeometry::CARDINAL_4_R150 ? "CB5" : "CB9")
                  << " raw:";
        for (int k = 0; k < 5; ++k) {
            std::cout << " p" << k << "=(" << fan.raw_points[k].x << "," << fan.raw_points[k].y
                      << ",c=" << fan.raw_confidences[k] << ")";
        }
        std::cout << "\n    six:";
        for (int k = 0; k < 6; ++k) {
            std::cout << " " << kpt_name(k) << "=(" << fan.points[k].x << "," << fan.points[k].y
                      << ")";
        }
        std::cout << "\n";

        RuneTarget::fan_pnp(fan, camera_info, ISO3::Identity(), true);
        std::cout << "    pnp_valid=" << fan.pose_valid;
        if (fan.pose_valid) {
            const auto t = fan.pose.translation();
            std::cout << " t=(" << t.x() << "," << t.y() << "," << t.z() << ")";
            auto object = RuneFanBladeWithR::Point3DRZERO<cv::Point3f>::build(fan.geometry);
            std::vector<cv::Point2f> img_points = fan.points;
            auto reproj = utils::reprojection(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                object,
                fan.pose
            );
            double sum = 0;
            for (size_t i = 0; i < reproj.size(); ++i) {
                sum += cv::norm(reproj[i] - img_points[i]);
            }
            std::cout << " reproj_mean_px=" << (sum / reproj.size());
            const auto r_reproj = utils::reprojection(
                camera_info.camera_matrix,
                camera_info.distortion_coefficients,
                { cv::Point3f(0, 0, 0) },
                fan.pose
            );
            std::cout << " r_err_px=" << cv::norm(r_reproj[0] - fan.points[RuneFanBladeWithR::R]);
            auto object6 = RuneFanBladeWithR::Point3DRZERO<cv::Point3f>::build(fan.geometry);
            try {
                auto pose6 = utils::solve_pnp(
                    fan.points,
                    object6,
                    camera_info.camera_matrix,
                    camera_info.distortion_coefficients
                );
                auto r6 = utils::reprojection(
                    camera_info.camera_matrix,
                    camera_info.distortion_coefficients,
                    { cv::Point3f(0, 0, 0) },
                    pose6
                );
                auto all6 = utils::reprojection(
                    camera_info.camera_matrix,
                    camera_info.distortion_coefficients,
                    object6,
                    pose6
                );
                double sum6 = 0;
                for (int i = 0; i < 6; ++i) sum6 += cv::norm(all6[i] - fan.points[i]);
                std::cout << " six_pnp_r_err_px=" << cv::norm(r6[0] - fan.points[RuneFanBladeWithR::R])
                          << " six_pnp_reproj_mean_px=" << (sum6 / 6.0);
            } catch (const std::exception& e) {
                std::cout << " six_pnp_error=" << e.what();
            }
        }
        std::cout << "\n";

        for (int k = 0; k < 5; ++k) {
            draw_point(vis, fan.raw_points[k], cv::Scalar(255, 0, 255), std::to_string(k));
        }
        for (int k = 0; k < 6; ++k) {
            draw_point(vis, fan.points[k], cv::Scalar(0, 255, 0), kpt_name(k));
        }
    }

    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
        const auto name = std::filesystem::path(image_path).filename().string();
        const auto out = std::filesystem::path(output_dir) / ("cb5_" + name);
        cv::imwrite(out.string(), vis);
        std::cout << "  wrote " << out << "\n";
    }
}

bool self_test() {
    using namespace awakening::auto_buff;
    YAML::Node cfg;
    cfg["model_type"] = "cb5";
    cfg["conf_threshold"] = 0.3;
    cfg["keypoint_conf_threshold"] = 0.3;
    cfg["nms_distance"] = 30.0;
    auto infer = RuneInfer::create(cfg);

    cv::Mat out = cv::Mat::zeros(18, 3, CV_32F);
    auto set_anchor = [&](int a, int cls, float class_conf, float x0, float y0) {
        out.at<float>(cls, a) = class_conf;
        for (int k = 0; k < 5; ++k) {
            const int base = 3 + k * 3;
            out.at<float>(base, a) = x0 + k * 10.0f;
            out.at<float>(base + 1, a) = y0 + k * 5.0f;
            out.at<float>(base + 2, a) = 0.95f;
        }
    };
    set_anchor(0, 1, 0.9f, 100.0f, 100.0f);
    set_anchor(1, 2, 0.8f, 400.0f, 400.0f);
    auto dets = infer->process(out);
    bool ok = dets.size() == 2;
    if (ok) {
        ok = dets[0].activation == RuneActivation::SMALL_ACTIVATED
            && dets[0].geometry == RuneBladeGeometry::CARDINAL_4_R150
            && dets[0].points.size() == 6
            && dets[0].points[RuneFanBladeWithR::R].x == 120.0f
            && dets[0].points[RuneFanBladeWithR::TOP].x == 100.0f
            && dets[0].points[RuneFanBladeWithR::LEFT].x == 110.0f
            && dets[0].points[RuneFanBladeWithR::RIGHT].x == 130.0f
            && dets[0].points[RuneFanBladeWithR::BOTTOM].x == 140.0f;
    }
    if (ok) {
        ok = dets[1].activation == RuneActivation::BIG_ACTIVATED;
    }
    // 输出通道不足时必须安全返回空结果，而不是抛异常/越界。
    cv::Mat bad = cv::Mat::zeros(10, 3, CV_32F);
    ok = ok && infer->process(bad).empty();
    std::cout << (ok ? "SELF_TEST PASS" : "SELF_TEST FAIL") << "\n";
    return ok;
}

struct BenchSummary {
    double avg_ms = 0.0;
    double p95_ms = 0.0;
    double fps = 0.0;
    double avg_objects = 0.0;
};

struct StageSamples {
    void add(const utils::NetDetectorBase::OutPut& output) {
        std::lock_guard<std::mutex> lock(mutex);
        preprocess.push_back(output.preprocess_ms);
        inference.push_back(output.inference_ms);
        output_copy.push_back(output.output_copy_ms);
    }
    std::mutex mutex;
    std::vector<double> preprocess;
    std::vector<double> inference;
    std::vector<double> output_copy;
};

std::pair<double, double> summarize_stage(std::vector<double> values) {
    if (values.empty()) {
        return { 0.0, 0.0 };
    }
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const size_t p95_index = static_cast<size_t>(0.95 * static_cast<double>(values.size() - 1));
    return { sum / static_cast<double>(values.size()), values[p95_index] };
}

template<typename Fn>
BenchSummary run_benchmark(Fn&& fn, int concurrency, int iterations, int warmup = 5) {
    std::vector<std::vector<double>> thread_latencies(static_cast<size_t>(concurrency));
    std::vector<size_t> thread_objects(static_cast<size_t>(concurrency), 0);
    std::barrier sync(concurrency + 1);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(concurrency));

    for (int worker = 0; worker < concurrency; ++worker) {
        workers.emplace_back([&, worker]() {
            for (int i = 0; i < warmup; ++i) {
                (void)fn();
            }
            sync.arrive_and_wait();
            sync.arrive_and_wait();

            auto& latencies = thread_latencies[static_cast<size_t>(worker)];
            latencies.reserve(static_cast<size_t>(iterations));
            for (int i = 0; i < iterations; ++i) {
                const auto start = Clock::now();
                thread_objects[static_cast<size_t>(worker)] += fn();
                const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start);
                latencies.push_back(elapsed.count());
            }
        });
    }

    sync.arrive_and_wait();
    const auto wall_start = Clock::now();
    sync.arrive_and_wait();
    for (auto& worker: workers) {
        worker.join();
    }
    const double wall_s = std::chrono::duration<double>(Clock::now() - wall_start).count();

    std::vector<double> latencies;
    latencies.reserve(static_cast<size_t>(concurrency * iterations));
    for (auto& local: thread_latencies) {
        latencies.insert(latencies.end(), local.begin(), local.end());
    }
    std::sort(latencies.begin(), latencies.end());
    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const size_t p95_index = static_cast<size_t>(0.95 * static_cast<double>(latencies.size() - 1));
    const size_t object_count =
        std::accumulate(thread_objects.begin(), thread_objects.end(), size_t { 0 });
    return BenchSummary {
        .avg_ms = sum / static_cast<double>(latencies.size()),
        .p95_ms = latencies[p95_index],
        .fps = static_cast<double>(latencies.size()) / wall_s,
        .avg_objects = static_cast<double>(object_count) / static_cast<double>(latencies.size()),
    };
}

void print_benchmark(const char* phase, const BenchSummary& result) {
    std::cout << std::fixed << std::setprecision(3) << phase << " avg_ms=" << result.avg_ms
              << " p95_ms=" << result.p95_ms << " fps=" << result.fps
              << " avg_objects=" << result.avg_objects << "\n";
}

utils::NetDetectorBase::Ptr make_net_detector(
    const YAML::Node& detector_config,
    const utils::NetDetectorBase::Config& net_config,
    const std::string& backend
) {
    if (backend == "openvino") {
        return std::make_unique<utils::NetDetectorOpenVINO>(
            detector_config["net_detector"]["openvino"],
            net_config
        );
    }
#ifdef USE_TRT
    if (backend == "tensorrt") {
        return std::make_unique<utils::NetDetectorTensorrt>(
            detector_config["net_detector"]["tensorrt"],
            net_config
        );
    }
#endif
    throw std::runtime_error("unsupported benchmark backend: " + backend);
}

utils::NetDetectorBase::Config make_net_config(const RuneInfer& infer) {
    return utils::NetDetectorBase::Config {
        .target_format = infer.target_format(),
        .preprocess_scale = infer.use_norm() ? 1.0 / 255.0 : 1.0,
        .target_w = infer.input_w(),
        .target_h = infer.input_h(),
    };
}

int benchmark(
    const std::string& config_path,
    const std::string& image_path,
    int concurrency,
    int iterations
) {
    if (concurrency < 1 || iterations < 1) {
        std::cerr << "concurrency and iterations must be positive\n";
        return 1;
    }
    const YAML::Node config = YAML::LoadFile(config_path);
    const YAML::Node detector_config = config["rune_detector"];
    cv::Mat src = cv::imread(image_path);
    if (src.empty()) {
        std::cerr << "cannot read image: " << image_path << "\n";
        return 1;
    }
    // Match the CB5 experiment camera resolution so preprocessing cost is representative.
    cv::resize(src, src, cv::Size(1440, 1080), 0, 0, cv::INTER_LINEAR);
    CommonFrame frame;
    frame.img_frame.src_img = src;
    frame.img_frame.format = PixelFormat::BGR;
    frame.img_frame.timestamp = Clock::now();

    std::cout << "benchmark config=" << config_path << " concurrency=" << concurrency
              << " iterations_per_worker=" << iterations << " image=" << src.cols << "x"
              << src.rows << "\n";

    auto infer = RuneInfer::create(detector_config["net_detector"]["rune_infer"]);
    const auto backend = detector_config["backend"].as<std::string>();
    auto net = make_net_detector(detector_config, make_net_config(*infer), backend);

    print_benchmark(
        "network_total",
        run_benchmark(
            [&]() {
                const auto output = net->detect(src, PixelFormat::BGR);
                return output.output.empty() ? size_t { 0 } : size_t { 1 };
            },
            concurrency,
            iterations
        )
    );

    StageSamples stage_samples;
    (void)run_benchmark(
        [&]() {
            const auto output = net->detect(src, PixelFormat::BGR);
            stage_samples.add(output);
            return output.output.empty() ? size_t { 0 } : size_t { 1 };
        },
        concurrency,
        iterations
    );
    const auto preprocess = summarize_stage(std::move(stage_samples.preprocess));
    const auto inference = summarize_stage(std::move(stage_samples.inference));
    const auto output_copy = summarize_stage(std::move(stage_samples.output_copy));
    std::cout << std::fixed << std::setprecision(3)
              << backend << "_stages preprocess_avg/p95_ms=" << preprocess.first << "/"
              << preprocess.second << " inference_avg/p95_ms=" << inference.first << "/"
              << inference.second << " output_copy_avg/p95_ms=" << output_copy.first << "/"
              << output_copy.second << "\n";

    const auto sample_output = net->detect(src, PixelFormat::BGR);
    print_benchmark(
        "cb5_postprocess",
        run_benchmark(
            [&]() { return infer->process(sample_output.output).size(); },
            concurrency,
            iterations
        )
    );

    RuneDetector detector(detector_config);
    print_benchmark(
        "detector_total",
        run_benchmark(
            [&]() {
                const auto detection = detector.detect(
                    frame,
                    cv::Rect(0, 0, src.cols, src.rows),
                    enemy_color_from_string(config["enemy_color"].as<std::string>())
                );
                return detection.fan_blades.size() + detection.fan_targets.size()
                    + detection.rune_rs.size();
            },
            concurrency,
            iterations
        )
    );
    return 0;
}

bool all_finite(const cv::Mat& output) {
    if (output.empty() || output.type() != CV_32F)
        return false;
    for (int row = 0; row < output.rows; ++row) {
        const float* values = output.ptr<float>(row);
        for (int col = 0; col < output.cols; ++col) {
            if (!std::isfinite(values[col]))
                return false;
        }
    }
    return true;
}

int compare_backends(const std::string& config_path, const std::string& image_path) {
#ifndef USE_TRT
    std::cerr << "TensorRT support is not enabled in this build\n";
    return 1;
#else
    const YAML::Node config = YAML::LoadFile(config_path);
    const YAML::Node detector_config = config["rune_detector"];
    cv::Mat src = cv::imread(image_path);
    if (src.empty()) {
        std::cerr << "cannot read image: " << image_path << "\n";
        return 1;
    }

    auto infer = RuneInfer::create(detector_config["net_detector"]["rune_infer"]);
    const auto net_config = make_net_config(*infer);
    auto openvino = make_net_detector(detector_config, net_config, "openvino");
    auto tensorrt = make_net_detector(detector_config, net_config, "tensorrt");

    // Exclude one-time backend initialization from the reported stage timings.
    (void)openvino->detect(src, PixelFormat::BGR);
    (void)tensorrt->detect(src, PixelFormat::BGR);
    const auto ov_output = openvino->detect(src, PixelFormat::BGR);
    const auto trt_output = tensorrt->detect(src, PixelFormat::BGR);
    const bool shapes_ok = ov_output.output.rows == 18 && ov_output.output.cols == 6300
        && trt_output.output.size() == ov_output.output.size();
    const bool finite_ok = all_finite(ov_output.output) && all_finite(trt_output.output);
    auto ov_objects = infer->process(ov_output.output);
    auto trt_objects = infer->process(trt_output.output);
    bool objects_ok = ov_objects.size() == trt_objects.size();
    bool pnp_ok = true;
    size_t valid_pnp_count = 0;
    float max_keypoint_delta = 0.0F;
    CameraInfo camera_info;
    camera_info.load(config["camera"]["camera_info"]);
    if (objects_ok) {
        for (size_t i = 0; i < ov_objects.size(); ++i) {
            auto& ov = ov_objects[i];
            auto& trt = trt_objects[i];
            objects_ok = objects_ok && ov.activation == trt.activation
                && ov.geometry == trt.geometry && ov.points.size() == trt.points.size();
            if (!objects_ok)
                break;
            for (size_t k = 0; k < ov.points.size(); ++k) {
                if (!std::isfinite(ov.points[k].x) || !std::isfinite(ov.points[k].y)
                    || !std::isfinite(trt.points[k].x) || !std::isfinite(trt.points[k].y)) {
                    objects_ok = false;
                    break;
                }
                max_keypoint_delta = std::max(
                    max_keypoint_delta,
                    static_cast<float>(cv::norm(ov.points[k] - trt.points[k]))
                );
            }
            ov.transform(ov_output.transform_matrix);
            trt.transform(trt_output.transform_matrix);
            RuneTarget::fan_pnp(ov, camera_info, ISO3::Identity(), true);
            RuneTarget::fan_pnp(trt, camera_info, ISO3::Identity(), true);
            pnp_ok = pnp_ok && ov.pose_valid == trt.pose_valid;
            if (ov.pose_valid && trt.pose_valid) {
                ++valid_pnp_count;
                pnp_ok = pnp_ok && ov.pose.translation().allFinite()
                    && trt.pose.translation().allFinite();
            }
        }
    }
    constexpr float MAX_REASONABLE_KEYPOINT_DELTA_PX = 5.0F;
    objects_ok = objects_ok && max_keypoint_delta <= MAX_REASONABLE_KEYPOINT_DELTA_PX;

    std::cout << std::fixed << std::setprecision(3)
              << "compare raw_shape openvino=" << ov_output.output.rows << "x"
              << ov_output.output.cols << " tensorrt=" << trt_output.output.rows << "x"
              << trt_output.output.cols << " finite=" << (finite_ok ? "yes" : "no") << "\n"
              << "compare objects openvino=" << ov_objects.size() << " tensorrt="
              << trt_objects.size() << " max_keypoint_delta_px=" << max_keypoint_delta
              << " valid_pnp_pairs=" << valid_pnp_count << "\n"
              << "compare timing_ms openvino(pre/infer/copy)=" << ov_output.preprocess_ms << "/"
              << ov_output.inference_ms << "/" << ov_output.output_copy_ms
              << " tensorrt(pre/infer/copy)=" << trt_output.preprocess_ms << "/"
              << trt_output.inference_ms << "/" << trt_output.output_copy_ms << "\n";
    const bool ok = shapes_ok && finite_ok && objects_ok && pnp_ok;
    std::cout << (ok ? "COMPARE PASS" : "COMPARE FAIL") << "\n";
    return ok ? 0 : 1;
#endif
}

int armor_smoke_test(const std::string& config_path, const std::string& image_path) {
    const YAML::Node config = YAML::LoadFile(config_path);
    cv::Mat src = cv::imread(image_path);
    if (src.empty()) {
        std::cerr << "cannot read image: " << image_path << "\n";
        return 1;
    }
    CommonFrame frame;
    frame.img_frame.src_img = src;
    frame.img_frame.format = PixelFormat::BGR;
    frame.img_frame.timestamp = Clock::now();
    auto_aim::ArmorDetector detector(config["armor_detector"]);
    auto [lights, armors] = detector.detect(frame, cv::Rect(0, 0, src.cols, src.rows));
    std::cout << "ARMOR_TRT_SMOKE PASS lights=" << lights.size() << " armors=" << armors.size()
              << "\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--self-test") {
        return self_test() ? 0 : 1;
    }
    if (argc >= 4 && std::string(argv[1]) == "--benchmark") {
        const int concurrency = argc >= 5 ? std::stoi(argv[4]) : 1;
        const int iterations = argc >= 6 ? std::stoi(argv[5]) : 50;
        try {
            return benchmark(argv[2], argv[3], concurrency, iterations);
        } catch (const std::exception& e) {
            std::cerr << "benchmark error: " << e.what() << "\n";
            return 1;
        }
    }
    if (argc == 4 && std::string(argv[1]) == "--compare") {
        try {
            return compare_backends(argv[2], argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "compare error: " << e.what() << "\n";
            return 1;
        }
    }
    if (argc == 4 && std::string(argv[1]) == "--armor-smoke") {
        try {
            return armor_smoke_test(argv[2], argv[3]);
        } catch (const std::exception& e) {
            std::cerr << "armor smoke error: " << e.what() << "\n";
            return 1;
        }
    }
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <config.yaml> <image-or-dir>... [--out <dir>]\n"
                  << "       " << argv[0] << " --compare <config.yaml> <image>\n"
                  << "       " << argv[0] << " --armor-smoke <config.yaml> <image>\n";
        return 1;
    }
    const std::string config_path = argv[1];
    std::string output_dir;
    std::vector<std::string> inputs;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) {
            output_dir = argv[++i];
        } else {
            inputs.push_back(arg);
        }
    }
    try {
        const YAML::Node config = YAML::LoadFile(config_path);
        CameraInfo camera_info;
        camera_info.load(config["camera"]["camera_info"]);
        const EnemyColor enemy_color = enemy_color_from_string(config["enemy_color"].as<std::string>());
        RuneDetector detector(config["rune_detector"]);
        for (const auto& input: inputs) {
            if (std::filesystem::is_directory(input)) {
                for (const auto& entry: std::filesystem::directory_iterator(input)) {
                    if (entry.is_regular_file()) {
                        process_image(
                            detector,
                            camera_info,
                            entry.path().string(),
                            output_dir,
                            enemy_color
                        );
                    }
                }
            } else {
                process_image(detector, camera_info, input, output_dir, enemy_color);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
