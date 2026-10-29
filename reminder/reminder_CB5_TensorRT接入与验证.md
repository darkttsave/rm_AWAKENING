# CB5 TensorRT 接入、验证与恢复

适用入口：`config/daedalus_sim_tensorrt_cb5.yaml`。本文只说明 CB5 能量机关 TensorRT 后端；OpenVINO CPU 调参见 `reminder_CB5_OpenVINO调参.md`。

## 1. 模型与配置配对

以下三项必须同时保持正确：

```yaml
rune_detector:
  backend: tensorrt
  net_detector:
    tensorrt:
      model_path: ${ROOT_DIR}/model/model-0624.onnx
      engine_path: ${ROOT_DIR}/model/model-0624_sm120.engine
    rune_infer:
      model_type: cb5
```

CB5 实测 I/O：

```text
images [1,3,480,640] -> output0 [1,18,6300]
```

不要将 `model-0624.onnx` 与 CB9 的 `model_type: cb` 配对，也不要让 CB5 共用装甲板或 CB9 engine 路径。

## 2. 构建与启动

首次配置示例（按本机安装位置调整 TensorRT 和 CUDA 路径）：

```bash
cmake -S . -B build-trt -G Ninja \
  -DBUILD_WITH_TRT=ON \
  -DBUILD_WITH_OPENVINO=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.9 \
  -DTensorRT_ROOT=/home/lzy/opt/TensorRT-10.13.3.9 \
  -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build build-trt -j2 --target standard rune_cb5_test
```

Daedalus 启动：

```bash
./bin/standard config/daedalus_sim_tensorrt_cb5.yaml true
```

同一套 Daedalus 共享内存不要同时运行两个 `standard`。切换 OpenVINO/TensorRT 前先停止旧实例，避免两个进程同时写控制结果。

如果运行时找不到 TensorRT、CUDA、OpenVINO 或 ROS 动态库，先按实际安装位置补充 `LD_LIBRARY_PATH`，不要把个人绝对路径写入主 YAML。

## 3. 关键参数

外部并发位于 `rune_detector.max_infer_num`；TensorRT 参数位于 `rune_detector.net_detector.tensorrt`。

| 参数 | 当前基线 | 含义与注意事项 |
|---|---|---|
| `max_infer_num` | `2` | 外部允许同时执行的能量机关检测数。当前队列满时仍沿用原有 skip 语义；不要借调参修改队列、Tracker 或 Aimer。 |
| `copy_context_num` | `2` | TensorRT execution context / CUDA stream 资源数。通常与 `max_infer_num` 对齐；小于外部并发会导致请求拿不到 Context，大于它只会增加显存占用。 |
| `min_free_mem_ratio` | `0.1` | 创建额外 Context 前要求保留的最小空闲显存比例。范围应在 0～1；只用于资源保护，不是性能旋钮。 |
| `use_cuda_preproces` | `true` | 使用 CUDA 完成 H2D、letterbox、BGR/RGB 交换和 NCHW 归一化。关闭后走 CPU/OpenCV 预处理，仅用于定位预处理差异。 |
| `engine_path` | `model/model-0624_sm120.engine` | CB5 独立 engine 缓存。文件与 GPU 架构、TensorRT、CUDA 版本绑定，已被 `.gitignore` 排除。 |

当前基线是两个外部检测槽位配两个 TensorRT Context。不要为不同 Context 数复制多份完整 YAML；一次只改一个参数并记录硬件、软件版本、输入频率和系统负载。

## 4. Engine 生成与恢复

首次运行没有缓存时，程序会从 ONNX 构建 FP16 engine。RTX 5060 Laptop、TensorRT 10.13.3.9 环境下首次 tactic 构建约需数分钟；构建期间没有帧输出属于正常现象。

成功日志应包含：

```text
Engine built & saved: .../model/model-0624_sm120.engine
TensorRT tensors: images [1,3,480,640] -> output0 [1,18,6300]
```

后续启动应看到：

```text
Loaded TensorRT engine: .../model/model-0624_sm120.engine
```

更换 GPU、CUDA 或 TensorRT 后，不要继续沿用旧缓存。恢复方式：

1. 停止 `standard`。
2. 将 `model/model-0624_sm120.engine` 改名为备份文件。
3. 确认 YAML 仍指向 `model/model-0624.onnx` 和独立 CB5 engine 路径。
4. 重新启动并等待 engine 构建完成。
5. 核对输入输出张量日志后再进行联调。

如果回退 OpenVINO，直接改用 `config/daedalus_sim_openvino_cb5.yaml`，不要修改 CB5 解析器或 Tracker 参数。

## 5. 独立验证

解析器自测：

```bash
./bin/rune_cb5_test --self-test
```

其中故意输入错误 shape 的测试会打印一条 shape mismatch error，最终出现 `SELF_TEST PASS` 即为通过。

同一张真实 CB5 图片的双后端对照：

```bash
./bin/rune_cb5_test --compare \
  config/daedalus_sim_tensorrt_cb5.yaml \
  /path/to/cb5-image.png
```

工具检查：

- OpenVINO/TensorRT 输出是否均为 `18 x 6300`；
- 是否存在 NaN/Inf；
- 目标数量、激活类别和关键点是否一致；
- 最大关键点偏差是否超过 5 px；
- 两端 PnP 的 pose-valid 是否一致，平移向量是否为有限值。

若图片没有检测目标，`COMPARE PASS` 只能证明空结果和张量通路正常，不能替代真实目标正确性验证。

双 Context 独立 benchmark：

```bash
./bin/rune_cb5_test --benchmark \
  config/daedalus_sim_tensorrt_cb5.yaml \
  /path/to/cb5-image.png 2 50
```

`tensorrt_stages` 中的 preprocess、inference、output-copy 由 CUDA event 测量，并在 GPU 完成后读取；不是异步提交耗时。聚合 FPS 是多个 worker 的总吞吐，不能当成单帧端到端延迟。

## 6. Daedalus 验证

启动后先确认日志明确显示加载的是 CB5 TensorRT engine，而不是 OpenVINO。随后读取每秒 `cb5_perf`：

- `camera / detected / skipped`；
- preprocess、infer、output-copy、postprocess 的 avg/P95；
- Detector、Detector→Tracker queue、Tracker、total 的 avg/P95；
- 是否出现空输出、CUDA/TensorRT 错误或崩溃。

比较时保持相近输入频率、场景和系统负载。首帧、engine 首次构建和预热阶段不要纳入稳定性能统计；旧 `avg_latency` 可能受队列批量出队影响，不能单独用于判断后端优劣。

### 2026-10-28 Daedalus 联调记录

日志：`2026-10-28_21-17-08.log`（日志文件本身不纳入 Git）。

- 启动日志确认加载 `model-0624_sm120.engine`，张量为 `[1,3,480,640] -> [1,18,6300]`。
- Human 汇总的 GPU inference 平均约 `0.811 ms`，Detector 平均约 `3.293 ms`。
- 稳定窗口中 detected 基本跟随 camera，除启动窗口出现 1 次 skipped 外未见持续积压。
- 运行正常停止，没有 TensorRT/CUDA 崩溃。

该记录证明当前硬件上的 TensorRT 后端、同步计时和端到端吞吐可用，但联调现场可能包含后续识别参数实验，不应把数值直接当作提交配置的识别效果基准，也不能替代真实 CB5 正样本的双后端一致性检查。

## 7. 当前验证边界

- 当前环境已完成 CB5 engine 构建/加载、张量核对、双 Context benchmark、CB5 parser self-test、CB9 TensorRT 和装甲板 TensorRT smoke。
- Human 已完成一次 Daedalus TensorRT 联调；GPU inference 平均约 0.811 ms、Detector 平均约 3.293 ms，推理基本跟上相机输入。
- 仓库内没有真实 CB5 目标图，仍需补一次有目标的 OpenVINO/TensorRT 对照和 PnP 检查。
- 当前结果是正确性与可运行性基线，不代表 TensorRT 已完成最终性能优化。
