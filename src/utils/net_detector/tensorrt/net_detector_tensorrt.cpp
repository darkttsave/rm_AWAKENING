#include <cstddef>
#ifdef USE_TRT
    #include "net_detector_tensorrt.hpp"
    #include "utils/buffer.hpp"
    #include "utils/common/image.hpp"
    #include "utils/cuda/letter_box.hpp"
    #include "utils/logger.hpp"
    #include <NvOnnxParser.h>
    #include <array>
    #include <chrono>
    #include <cuda_runtime.h>
    #include <fstream>
    #include <memory>
    #include <opencv2/core/hal/interface.h>
    #include <opencv2/highgui.hpp>
    #include <sstream>
    #include <string>
namespace awakening::utils {
    #define TRT_CHECK(expr) \
        do { \
            const auto _ret = (expr); \
            if (_ret != cudaSuccess) { \
                std::cerr << "\033[31mCUDA error: " << cudaGetErrorString(_ret) << " (" << #expr \
                          << ")\033[0m\n"; \
                std::abort(); \
            } \
        } while (0)
struct NetDetectorTensorrt::Impl {
    struct Params {
        std::string model_path;
        std::string engine_path;
        int copy_context_num = 1;
        double min_free_mem_ratio = 0.1;
        bool use_cuda_preproces = true;
        void load(const YAML::Node& config) {
            model_path = replace_root_dir(config["model_path"].as<std::string>());
            if (config["engine_path"]) {
                engine_path = replace_root_dir(config["engine_path"].as<std::string>());
            }
            copy_context_num = config["copy_context_num"].as<int>();
            min_free_mem_ratio = config["min_free_mem_ratio"].as<double>();
            use_cuda_preproces = config["use_cuda_preproces"].as<bool>();
        }
    } params_;
    class TRTLogger: public nvinfer1::ILogger {
    public:
        explicit TRTLogger(
            nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING
        ):
            severity_(severity) {}
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
            if (severity <= severity_) {
                std::cerr << msg << std::endl;
            }
        }
        nvinfer1::ILogger::Severity severity_;
    };
    static int64_t volume(const nvinfer1::Dims& dims) {
        int64_t v = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0)
                throw std::runtime_error("Tensor shape is not fully specified");
            v *= dims.d[i];
        }
        return v;
    }
    static bool sameDims(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) {
        if (lhs.nbDims != rhs.nbDims)
            return false;
        for (int i = 0; i < lhs.nbDims; ++i) {
            if (lhs.d[i] != rhs.d[i])
                return false;
        }
        return true;
    }
    static bool hasDynamicDims(const nvinfer1::Dims& dims) {
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] < 0)
                return true;
        }
        return false;
    }
    static std::string dimsString(const nvinfer1::Dims& dims) {
        std::ostringstream oss;
        oss << "[";
        for (int i = 0; i < dims.nbDims; ++i) {
            if (i > 0)
                oss << ",";
            oss << dims.d[i];
        }
        oss << "]";
        return oss.str();
    }
    Impl(const YAML::Node& config, Config c) {
        config_ = c;

        params_.load(config);
        buildEngine(params_.model_path, params_.engine_path);
        auto tmp_ctx =
            std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());

        if (!tmp_ctx)
            throw std::runtime_error("Failed to create execution context");
        if (engine_->getNbIOTensors() != 2)
            throw std::runtime_error("TensorRT adapter requires exactly one input and one output");
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
                input_name_ = name;
            else
                output_name_ = name;
        }
        if (!input_name_ || !output_name_)
            throw std::runtime_error("TensorRT input/output tensor discovery failed");
        if (engine_->getTensorDataType(input_name_) != nvinfer1::DataType::kFLOAT
            || engine_->getTensorDataType(output_name_) != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("TensorRT adapter currently requires FP32 input/output tensors");
        input_dims_ = nvinfer1::Dims4 { 1,
                                        config_.target_format == PixelFormat::GRAY ? 1 : 3,
                                        config_.target_h,
                                        config_.target_w };
        const auto engine_input_dims = engine_->getTensorShape(input_name_);
        if (hasDynamicDims(engine_input_dims)) {
            if (!tmp_ctx->setInputShape(input_name_, input_dims_))
                throw std::runtime_error("Failed to set TensorRT input shape");
        } else if (!sameDims(engine_input_dims, input_dims_)) {
            throw std::runtime_error(
                "TensorRT input shape mismatch: engine=" + dimsString(engine_input_dims)
                + " requested=" + dimsString(input_dims_)
            );
        }
        auto dims = tmp_ctx->getTensorShape(input_name_);
        input_dims_ = dims;
        output_dims_ = tmp_ctx->getTensorShape(output_name_);
        if (output_dims_.nbDims != 3 || output_dims_.d[0] != 1)
            throw std::runtime_error(
                "TensorRT adapter requires a three-dimensional batch-1 output, got "
                + dimsString(output_dims_)
            );
        tmp_ctx.reset();
        input_sz_ = volume(input_dims_);
        output_sz_ = volume(output_dims_);
        AWAKENING_INFO(
            "TensorRT tensors: {} {} -> {} {}",
            input_name_,
            dimsString(input_dims_),
            output_name_,
            dimsString(output_dims_)
        );
        if (params_.copy_context_num < 1) {
            params_.copy_context_num = 1;
        }
        for (int i = 0; i < params_.copy_context_num; ++i) {
            if (i > 0) {
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);

                AWAKENING_DEBUG("Free GPU memory: {} MB", free_mem / 1024.0 / 1024.0);
                AWAKENING_DEBUG("Total GPU memory: {} MB", total_mem / 1024.0 / 1024.0);
                double free_mem_ratio =
                    static_cast<double>(free_mem) / static_cast<double>(total_mem);
                if (free_mem_ratio < params_.min_free_mem_ratio && i > 0) {
                    AWAKENING_WARN(
                        "GPU memory is not enough! Free GPU memory: {:.2f}%",
                        free_mem_ratio * 100
                    );
                    break;
                }
            }

            Ctx ctx;
            ctx.context.reset(engine_->createExecutionContext());
            if (params_.use_cuda_preproces)
                ctx.letter_box = std::make_shared<__cuda::LetterBox>(config_);
            if (!params_.use_cuda_preproces)
                TRT_CHECK(cudaMalloc(&ctx.device_buffers[INPUT_IDX], input_sz_ * sizeof(float)));
            TRT_CHECK(cudaMalloc(&ctx.device_buffers[OUTPUT_IDX], output_sz_ * sizeof(float)));
            ctx.output_buffer.resize(output_sz_);
            TRT_CHECK(cudaStreamCreate(&ctx.stream));
            ctx_buffers_.add_resource(std::move(ctx));
        }
    }
    void buildEngine(const std::string& onnx_path, const std::string& configured_engine_path) {
        // Keep the historical ONNX-derived name when engine_path is omitted, but
        // allow a per-model engine name so models such as 1208 do not collide with
        // legacy engines checked into the repository.
        const std::string engine_path = configured_engine_path.empty()
            ? onnx_path.substr(0, onnx_path.find_last_of('.')) + ".engine"
            : configured_engine_path;

        runtime_.reset(nvinfer1::createInferRuntime(g_logger_));

        {
            std::ifstream fin(engine_path, std::ios::binary);
            if (fin.good()) {
                fin.seekg(0, std::ios::end);
                const size_t size = fin.tellg();
                fin.seekg(0, std::ios::beg);

                std::vector<char> data(size);
                fin.read(data.data(), size);

                engine_.reset(runtime_->deserializeCudaEngine(data.data(), size));
                if (engine_) {
                    AWAKENING_INFO("Loaded TensorRT engine: {}", engine_path);
                    return;
                }
                AWAKENING_ERROR("Failed to load TensorRT engine: {}", engine_path);
            }
        }

        AWAKENING_INFO("Building TensorRT engine from ONNX...");
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger_));

        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));

        auto parser =
            std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g_logger_));

        if (!parser->parseFromFile(
                onnx_path.c_str(),
                static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
            ))
        {
            AWAKENING_ERROR("Failed to parse ONNX: {}", onnx_path);
            throw std::runtime_error("Failed to parse ONNX: " + onnx_path);
        }

        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
        int c = config_.target_format == PixelFormat::GRAY ? 1 : 3;
        nvinfer1::Dims4 dims { 1, c, config_.target_h, config_.target_w };
        if (network->getNbInputs() != 1 || network->getNbOutputs() != 1)
            throw std::runtime_error("TensorRT adapter requires exactly one input and one output");
        auto* input = network->getInput(0);
        const auto model_dims = input->getDimensions();
        if (hasDynamicDims(model_dims)) {
            auto* profile = builder->createOptimizationProfile();
            const char* input_name = input->getName();
            if (!profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, dims)
                || !profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, dims)
                || !profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, dims)
                || config->addOptimizationProfile(profile) < 0)
                throw std::runtime_error("Failed to configure TensorRT optimization profile");
        } else if (!sameDims(model_dims, dims)) {
            throw std::runtime_error(
                "ONNX input shape mismatch: model=" + dimsString(model_dims)
                + " requested=" + dimsString(dims)
            );
        }

        auto serialized = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config)
        );
        if (!serialized)
            throw std::runtime_error("TensorRT engine serialization failed");

        engine_.reset(runtime_->deserializeCudaEngine(serialized->data(), serialized->size()));

        if (!engine_)
            throw std::runtime_error("Engine build failed");

        std::ofstream fout(engine_path, std::ios::binary);
        fout.write(static_cast<const char*>(serialized->data()), serialized->size());
        if (!fout.good())
            throw std::runtime_error("Failed to save TensorRT engine: " + engine_path);

        AWAKENING_INFO("Engine built & saved: {}", engine_path);
    }

    struct CudaStageTimer {
        std::array<cudaEvent_t, 4> events {};
        CudaStageTimer() {
            for (auto& event: events)
                TRT_CHECK(cudaEventCreate(&event));
        }
        ~CudaStageTimer() {
            for (auto event: events) {
                if (event)
                    cudaEventDestroy(event);
            }
        }
        void record(size_t index, cudaStream_t stream) {
            TRT_CHECK(cudaEventRecord(events[index], stream));
        }
        double elapsed(size_t begin, size_t end) const {
            float ms = 0.0F;
            TRT_CHECK(cudaEventElapsedTime(&ms, events[begin], events[end]));
            return ms;
        }
    };

    OutPut detect(const cv::Mat& img, PixelFormat format) noexcept {
        if (img.empty()) {
            return {};
        }
        OutPut output;
        cv::Mat blob;
        double cpu_preprocess_ms = 0.0;
        if (!params_.use_cuda_preproces) { // 最大化ctx利用率,该部分不需要ctx则暂时不请求c
            const auto preprocess_start = std::chrono::steady_clock::now();
            output.resized_img =
                utils::letterbox(img, output.transform_matrix, config_.target_w, config_.target_h);
            auto swap_rb = format != config_.target_format;
            blob = cv::dnn::blobFromImage(
                output.resized_img,
                config_.preprocess_scale,
                cv::Size(config_.target_w, config_.target_h),
                cv::Scalar(0, 0, 0),
                swap_rb
            );
            cpu_preprocess_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - preprocess_start
            )
                                    .count();
        }
        {
            auto r = ctx_buffers_.acquire();
            if (!r) {
                return output;
            }
            auto& ctx = *r;
            CudaStageTimer timer;
            timer.record(0, ctx.stream);
            if (params_.use_cuda_preproces) {
                auto tensor = ctx.letter_box->letterbox_pitched(
                    img.data,
                    format,
                    img.cols,
                    img.rows,
                    img.step,
                    output.transform_matrix,
                    ctx.stream
                );
                if (!tensor) {
                    return output;
                }
                ctx.device_buffers[INPUT_IDX] = tensor;
                output.resized_img = ctx.letter_box->tensor_to_mat(
                    static_cast<float*>(ctx.device_buffers[INPUT_IDX]),
                    ctx.stream,
                    format != config_.target_format
                );
            } else {
                TRT_CHECK(cudaMemcpyAsync(
                    ctx.device_buffers[INPUT_IDX],
                    blob.ptr<float>(),
                    input_sz_ * sizeof(float),
                    cudaMemcpyHostToDevice,
                    ctx.stream
                ));
            }
            timer.record(1, ctx.stream);

            ctx.context->setTensorAddress(input_name_, ctx.device_buffers[INPUT_IDX]);
            ctx.context->setTensorAddress(output_name_, ctx.device_buffers[OUTPUT_IDX]);

            if (!ctx.context->enqueueV3(ctx.stream)) {
                AWAKENING_ERROR("enqueueV3 failed");
                return output;
            }
            timer.record(2, ctx.stream);

            TRT_CHECK(cudaMemcpyAsync(
                ctx.output_buffer.data(),
                ctx.device_buffers[OUTPUT_IDX],
                output_sz_ * sizeof(float),
                cudaMemcpyDeviceToHost,
                ctx.stream
            ));
            timer.record(3, ctx.stream);

            TRT_CHECK(cudaEventSynchronize(timer.events[3]));
            output.preprocess_ms = cpu_preprocess_ms + timer.elapsed(0, 1);
            output.inference_ms = timer.elapsed(1, 2);
            output.output_copy_ms = timer.elapsed(2, 3);

            output.output =
                cv::Mat(output_dims_.d[1], output_dims_.d[2], CV_32F, ctx.output_buffer.data())
                    .clone();
        }

        return output;
    }
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    TRTLogger g_logger_;
    struct Ctx {
        std::shared_ptr<nvinfer1::IExecutionContext> context;
        std::array<void*, 2> device_buffers;
        std::vector<float> output_buffer;
        cudaStream_t stream { nullptr };
        __cuda::LetterBox::Ptr letter_box;
    };
    ResourcePool<Ctx> ctx_buffers_;
    constexpr static size_t INPUT_IDX = 0;
    constexpr static size_t OUTPUT_IDX = 1;

    size_t input_sz_ { 0 }, output_sz_ { 0 };

    nvinfer1::Dims input_dims_ {};
    nvinfer1::Dims output_dims_ {};
    const char* input_name_ { nullptr };
    const char* output_name_ { nullptr };
    Config config_;
};
NetDetectorTensorrt::NetDetectorTensorrt(const YAML::Node& config, Config c) {
    _impl = std::make_unique<NetDetectorTensorrt::Impl>(config, c);
}
NetDetectorTensorrt::~NetDetectorTensorrt() noexcept {
    _impl.reset();
}
NetDetectorTensorrt::OutPut
NetDetectorTensorrt::detect(const cv::Mat& img, PixelFormat format) noexcept {
    return _impl->detect(img, format);
}
} // namespace awakening::utils
#endif
