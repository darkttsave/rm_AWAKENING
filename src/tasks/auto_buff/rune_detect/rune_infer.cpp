#include "rune_infer.hpp"
#include "tasks/auto_buff/type.hpp"
#include "utils/logger.hpp"
#include "utils/utils.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace awakening::auto_buff {
static constexpr float MERGE_CONF_ERROR = 0.95f;
static constexpr float MERGE_MIN_IOU = 0.9f;
static constexpr float NMS_THRESHOLD = 0.35;
static constexpr int TOP_K = 128;
enum class Mode : int { CB, CB5 };
inline Mode modeFromString(const std::string& s) noexcept {
    std::string str = utils::to_upper(s);
    if (str == "CB5" || str == "SZU5" || str == "FIVE")
        return Mode::CB5;
    return Mode::CB;
}
template<Mode M>
struct ModelTraits; // declare
// TUP / 原九点模型
template<>
struct ModelTraits<Mode::CB> {
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr bool USE_NORM = true;
    static constexpr int NUM_POINTS = 9; // raw model output keypoints
    static constexpr int REMAPPED_KPTS = 6; // remapped to old 6-pt layout for solver compatibility
    static constexpr int NUM_CLASSES = 2;
    static constexpr int KPT_DIM = 3; // x, y, visibility per keypoint
    static constexpr int KPT_START = 6; // keypoints start at row 6 (after 4 bbox + 2 class)
    static constexpr PixelFormat TARGET_FORMAT = PixelFormat::RGB;
};
// 深圳大学 model-0624：3 类激活状态 + 5 关键点，输出 [1,18,6300]。
template<>
struct ModelTraits<Mode::CB5> {
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 480;
    static constexpr bool USE_NORM = true;
    static constexpr int NUM_POINTS = 5;
    static constexpr int REMAPPED_KPTS = 6;
    static constexpr int NUM_CLASSES = 3;
    static constexpr int KPT_DIM = 3;
    static constexpr int KPT_START = 3; // 3 class scores, then 5 * (x, y, conf)
    static constexpr int R_INDEX = 2; // 实测 model-0624：kpt[2] 是 R 标中心
    static constexpr PixelFormat TARGET_FORMAT = PixelFormat::RGB;
};

inline void nms_merge_sorted_bboxes(
    std::vector<RuneFanBladeWithR>& objs,
    std::vector<int>& out_indices,
    float nms_threshold
) {
    out_indices.clear();
    const size_t n = objs.size();

    for (size_t i = 0; i < n; ++i) {
        RuneFanBladeWithR& a = objs[i];
        bool keep = true;
        for (int idx: out_indices) {
            RuneFanBladeWithR& b = objs[idx];
            const float iou = utils::rect_ioU(a.bbox, b.bbox);
            if (std::isnan(iou) || iou > nms_threshold) {
                keep = false;
                if (a.color == b.color && iou > MERGE_MIN_IOU
                    && std::abs(a.confidence - b.confidence) < MERGE_CONF_ERROR)
                {
                    b.tmp_points.push_back(a.points);
                }
                break;
            }
        }
        if (keep)
            out_indices.push_back(static_cast<int>(i));
    }
}

inline std::vector<RuneFanBladeWithR> topk_and_nms(std::vector<RuneFanBladeWithR>& objs) {
    std::sort(objs.begin(), objs.end(), [](const RuneFanBladeWithR& a, const RuneFanBladeWithR& b) {
        return a.confidence > b.confidence;
    });

    if (static_cast<int>(objs.size()) > TOP_K)
        objs.resize(static_cast<size_t>(TOP_K));

    std::vector<int> indices;
    nms_merge_sorted_bboxes(objs, indices, NMS_THRESHOLD);

    std::vector<RuneFanBladeWithR> result;
    result.reserve(indices.size());

    for (size_t i = 0; i < indices.size(); ++i) {
        result.push_back(std::move(objs[indices[i]]));
        auto& ro = result.back();
        if (ro.tmp_points.size() >= 1) {
            std::vector<cv::Point2f> accum(ro.points.size(), cv::Point2f(0.0f, 0.0f));
            std::vector<int> count(ro.points.size(), 0);
            for (const auto& pts: ro.tmp_points) {
                if (pts.size() != ro.points.size())
                    continue;
                for (size_t k = 0; k < pts.size(); ++k) {
                    accum[k] += pts[k];
                    count[k]++;
                }
            }
            for (size_t k = 0; k < ro.points.size(); ++k) {
                if (count[k] > 0) {
                    ro.points[k] = (ro.points[k] + accum[k]) / static_cast<float>(count[k] + 1);
                }
            }
            ro.tmp_points.clear();
        }
    }

    return result;
}

// CB5 使用中心点距离 NMS，而不是 BBox IoU（模型不输出独立 bbox）。
inline std::vector<RuneFanBladeWithR>
nms_by_center(std::vector<RuneFanBladeWithR> objs, float distance_threshold) {
    std::sort(objs.begin(), objs.end(), [](const RuneFanBladeWithR& a, const RuneFanBladeWithR& b) {
        return a.confidence > b.confidence;
    });
    if (static_cast<int>(objs.size()) > TOP_K)
        objs.resize(static_cast<size_t>(TOP_K));

    const float d2 = distance_threshold * distance_threshold;
    std::vector<RuneFanBladeWithR> result;
    result.reserve(objs.size());
    for (auto& obj: objs) {
        const cv::Point2f center = obj.points[RuneFanBladeWithR::CENTER];
        bool keep = true;
        for (const auto& kept: result) {
            const cv::Point2f diff = center - kept.points[RuneFanBladeWithR::CENTER];
            if (diff.dot(diff) < d2) {
                keep = false;
                break;
            }
        }
        if (keep)
            result.push_back(std::move(obj));
    }
    return result;
}

struct RuneInfer::Impl {
    struct Params {
        double conf_threshold = 0.1;
        double keypoint_conf_threshold = 0.3;
        double nms_distance = 30.0;
        int min_valid_kpts = 4;
        Mode mode = Mode::CB;
        int input_w = ModelTraits<Mode::CB>::INPUT_W;
        int input_h = ModelTraits<Mode::CB>::INPUT_H;
        bool use_norm = ModelTraits<Mode::CB>::USE_NORM;
        PixelFormat target_format = ModelTraits<Mode::CB>::TARGET_FORMAT;
        template<typename M>
        void set_mode() {
            input_w = M::INPUT_W;
            input_h = M::INPUT_H;
            use_norm = M::USE_NORM;
            target_format = M::TARGET_FORMAT;
        }
        void load(const YAML::Node& config) {
            auto mode_str = config["model_type"].as<std::string>();
            mode = modeFromString(mode_str);
            switch (mode) {
                case Mode::CB: {
                    set_mode<ModelTraits<Mode::CB>>();
                    break;
                }
                case Mode::CB5: {
                    set_mode<ModelTraits<Mode::CB5>>();
                    break;
                }
            }
            conf_threshold = config["conf_threshold"].as<double>();
            if (config["keypoint_conf_threshold"]) {
                keypoint_conf_threshold = config["keypoint_conf_threshold"].as<double>();
            }
            if (config["nms_distance"]) {
                nms_distance = config["nms_distance"].as<double>();
            }
            if (config["min_valid_kpts"]) {
                min_valid_kpts = config["min_valid_kpts"].as<int>();
            }
        }
    } params_;
    Impl(const YAML::Node& config) {
        params_.load(config);
    }

    [[nodiscard]] std::vector<RuneFanBladeWithR> process(const cv::Mat& output_buffer) const {
        try {
            auto results = post_process(output_buffer);
            switch (params_.mode) {
                case Mode::CB:
                    // 保持原九点模式的后处理路径不变。
                    return topk_and_nms(results);
                case Mode::CB5:
                    return nms_by_center(std::move(results), static_cast<float>(params_.nms_distance));
            }
        } catch (const std::exception& e) {
            AWAKENING_ERROR("rune infer postprocess failed: {}", e.what());
        } catch (...) {
            AWAKENING_ERROR("rune infer postprocess failed: unknown exception");
        }
        return {};
    }
    [[nodiscard]] std::vector<RuneFanBladeWithR> post_process(const cv::Mat& output_buffer) const {
        if (output_buffer.empty())
            return {};
        switch (params_.mode) {
            case Mode::CB:
                return post_processCB_impl(output_buffer);
            case Mode::CB5:
                return post_processCB5_impl(output_buffer);
        }
        return {};
    }
    std::vector<RuneFanBladeWithR> post_processCB_impl(const cv::Mat& out) const {
        std::vector<RuneFanBladeWithR> results;
        const int cols = out.cols;
        auto remap_keypoints = [](const std::vector<cv::Point2f>& raw) {
            // Raw 9-pt: [0]=R_center, [1,8]=bottom, [2,3]=right, [4,5]=top, [6,7]=left
            // Compute edge-pair midpoints → map to old 6-pt PnP corners
            const cv::Point2f top = (raw[4] + raw[5]) * 0.5f;
            const cv::Point2f left = (raw[6] + raw[7]) * 0.5f;
            const cv::Point2f bottom = (raw[1] + raw[8]) * 0.5f;
            const cv::Point2f right = (raw[2] + raw[3]) * 0.5f;

            std::vector<cv::Point2f> out(ModelTraits<Mode::CB>::REMAPPED_KPTS);
            out[0] = top; // top edge center
            out[1] = left; // left edge center
            out[2] = bottom; // bottom edge center
            out[3] = right; // right edge center
            out[4] = (top + left + bottom + right) * 0.25f; // blade center
            out[5] = raw[0]; // R center
            return out;
        };
        for (int i = 0; i < cols; ++i) {
            const float r_score = out.at<float>(4, i);
            const float b_score = out.at<float>(5, i);
            float conf = -1;
            bool is_r = false;
            if (r_score > b_score) {
                is_r = true;
                conf = r_score;
            } else {
                is_r = false;
                conf = b_score;
            }
            if (conf < params_.conf_threshold)
                continue;
            const float cx = out.at<float>(0, i);
            const float cy = out.at<float>(1, i);
            const float ow = out.at<float>(2, i);
            const float oh = out.at<float>(3, i);
            RuneFanBladeWithR rune;
            rune.confidence = conf;
            rune.color = is_r ? RuneColor::RED : RuneColor::BLUE;
            rune.bbox = cv::Rect2f(cx - ow / 2, cy - oh / 2, ow, oh);
            cv::Mat kpts = out.col(i).rowRange(
                ModelTraits<Mode::CB>::KPT_START,
                ModelTraits<Mode::CB>::KPT_START
                    + (ModelTraits<Mode::CB>::NUM_POINTS * ModelTraits<Mode::CB>::KPT_DIM)
            );
            std::vector<cv::Point2f> raw(ModelTraits<Mode::CB>::NUM_POINTS);
            for (int j = 0; j < ModelTraits<Mode::CB>::NUM_POINTS; ++j) {
                raw[j] = cv::Point2f(
                    kpts.at<float>(j * ModelTraits<Mode::CB>::KPT_DIM, 0),
                    kpts.at<float>(j * ModelTraits<Mode::CB>::KPT_DIM + 1, 0)
                );
            }
            rune.points = remap_keypoints(raw);
            // rune.points = raw;
            results.push_back(rune);
        }
        return topk_and_nms(results);
    }

    // model-0624: 3 class scores + 5 keypoints * (x, y, conf) = 18 channels.
    std::vector<RuneFanBladeWithR> post_processCB5_impl(const cv::Mat& out) const {
        constexpr int channels = ModelTraits<Mode::CB5>::NUM_CLASSES
            + ModelTraits<Mode::CB5>::NUM_POINTS * ModelTraits<Mode::CB5>::KPT_DIM;
        if (out.empty() || out.type() != CV_32F) {
            AWAKENING_ERROR("CB5 output is empty or not CV_32F");
            return {};
        }
        // 同时兼容 [C, A] 和 [A, C] 两种导出布局。
        const bool channel_first = out.rows == channels;
        const bool channel_last = out.cols == channels;
        if (!channel_first && !channel_last) {
            AWAKENING_ERROR(
                "CB5 output shape mismatch: {}x{}, expected {} channels",
                out.rows,
                out.cols,
                channels
            );
            return {};
        }
        const int anchors = channel_first ? out.cols : out.rows;
        const auto at = [&](int c, int i) -> float {
            return channel_first ? out.at<float>(c, i) : out.at<float>(i, c);
        };

        std::vector<RuneFanBladeWithR> results;
        results.reserve(64);
        for (int i = 0; i < anchors; ++i) {
            int best_cls = -1;
            float class_conf = 0.0f;
            for (int c = 0; c < ModelTraits<Mode::CB5>::NUM_CLASSES; ++c) {
                const float score = at(c, i);
                if (std::isfinite(score) && score > class_conf) {
                    class_conf = score;
                    best_cls = c;
                }
            }
            if (best_cls < 0 || class_conf < params_.conf_threshold)
                continue;

            std::array<cv::Point2f, ModelTraits<Mode::CB5>::NUM_POINTS> raw;
            std::array<float, ModelTraits<Mode::CB5>::NUM_POINTS> raw_conf {};
            int valid_kpts = 0;
            for (int k = 0; k < ModelTraits<Mode::CB5>::NUM_POINTS; ++k) {
                const int base = ModelTraits<Mode::CB5>::KPT_START
                    + k * ModelTraits<Mode::CB5>::KPT_DIM;
                const float x = at(base, i);
                const float y = at(base + 1, i);
                const float kconf = at(base + 2, i);
                raw[k] = cv::Point2f(x, y);
                raw_conf[k] = kconf;
                if (std::isfinite(x) && std::isfinite(y) && std::isfinite(kconf)
                    && kconf >= params_.keypoint_conf_threshold)
                {
                    ++valid_kpts;
                }
            }
            constexpr int r_index = ModelTraits<Mode::CB5>::R_INDEX;
            const bool r_valid = std::isfinite(raw[r_index].x)
                && std::isfinite(raw[r_index].y)
                && raw_conf[r_index] >= params_.keypoint_conf_threshold;
            bool corners_valid = true;
            for (int k: { 0, 1, 3, 4 }) {
                corners_valid = corners_valid && std::isfinite(raw[k].x)
                    && std::isfinite(raw[k].y)
                    && raw_conf[k] >= params_.keypoint_conf_threshold;
            }
            // R 必须有效；四角也必须全部有效，否则不把无效点送入 PnP。
            if (valid_kpts < std::clamp(params_.min_valid_kpts, 4, 5) || !r_valid
                || !corners_valid)
            {
                continue;
            }

            // 官方标注顺序（实测）：
            //   0 = 上, 1 = 左, 2 = R, 3 = 右, 4 = 下。
            // 下游六点顺序是上/左/下/右/中/R，因此 3 与 4 需要交换。
            RuneFanBladeWithR rune;
            rune.geometry = RuneBladeGeometry::CARDINAL_4_R150;
            rune.activation = static_cast<RuneActivation>(best_cls);
            rune.confidence = class_conf;
            rune.color = RuneColor::NONE; // 类别是激活状态，不是红蓝颜色
            rune.raw_points = raw;
            rune.raw_confidences = raw_conf;
            rune.points.resize(RuneFanBladeWithR::N);
            rune.points[RuneFanBladeWithR::TOP] = raw[0];
            rune.points[RuneFanBladeWithR::LEFT] = raw[1];
            rune.points[RuneFanBladeWithR::BOTTOM] = raw[4];
            rune.points[RuneFanBladeWithR::RIGHT] = raw[3];
            rune.points[RuneFanBladeWithR::CENTER] =
                (raw[0] + raw[1] + raw[3] + raw[4]) * 0.25f;
            rune.points[RuneFanBladeWithR::R] = raw[r_index];

            const float min_x = std::min({ raw[0].x, raw[1].x, raw[3].x, raw[4].x });
            const float max_x = std::max({ raw[0].x, raw[1].x, raw[3].x, raw[4].x });
            const float min_y = std::min({ raw[0].y, raw[1].y, raw[3].y, raw[4].y });
            const float max_y = std::max({ raw[0].y, raw[1].y, raw[3].y, raw[4].y });
            rune.bbox = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
            results.push_back(rune);
        }
        return results;
    }

    int input_w() const noexcept {
        return params_.input_w;
    }
    int input_h() const noexcept {
        return params_.input_h;
    }
    bool use_norm() const noexcept {
        return params_.use_norm;
    }
    PixelFormat target_format() const noexcept {
        return params_.target_format;
    }
};

RuneInfer::RuneInfer(const YAML::Node& config) {
    _impl = std::make_unique<Impl>(config);
}
RuneInfer::~RuneInfer() noexcept {
    _impl.reset();
}
[[nodiscard]] std::vector<RuneFanBladeWithR> RuneInfer::process(const cv::Mat& output_buffer
) const {
    return _impl->process(output_buffer);
}

int RuneInfer::input_w() const noexcept {
    return _impl->input_w();
}
int RuneInfer::input_h() const noexcept {
    return _impl->input_h();
}
bool RuneInfer::use_norm() const noexcept {
    return _impl->use_norm();
}
PixelFormat RuneInfer::target_format() const noexcept {
    return _impl->target_format();
}
} // namespace awakening::auto_buff
