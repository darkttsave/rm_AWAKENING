# CB5 OpenVINO 调参与恢复

适用入口：`config/daedalus_sim_openvino_cb5.yaml`。以下参数与结论用于 CB5 OpenVINO CPU 后端，不应直接照搬到其他模型或硬件。

## 1. 启动

```bash
cmake --build build-trt -j2 --target standard rune_cb5_test

LD_LIBRARY_PATH=/opt/ros/jazzy/lib ./bin/standard \
  config/daedalus_sim_openvino_cb5.yaml false
```

主配置的 CB5 配对必须保持：

```yaml
model_path: ${ROOT_DIR}/model/model-0624.onnx
model_type: cb5
```

## 2. 参数位置与调整原则

外部并发位于 `rune_detector.max_infer_num`；其余参数位于 `rune_detector.net_detector.openvino`。

| 参数 | 当前基线 | 含义与合理调整范围 |
|---|---|---|
| `max_infer_num` | `2` | 外部同时执行检测的最大数量。通常测试 1～2；增大前必须同时验证 skip、CPU 占用和尾延迟。槽位满时当前语义是跳过网络检测，不要借调参改变队列或 Tracker 行为。 |
| `perf_mode` | `LATENCY` | OpenVINO 性能提示。低延迟任务使用 `LATENCY`；`THROUGHPUT` 仅在吞吐优先场景单独验证。 |
| `inference_num_threads` | 省略（自动） | 推理最大线程数，必须为正整数。优先省略；需要限制资源时，从不超过逻辑 CPU 数的少量代表值开始测试，例如 4、8、12，不能跨硬件照搬。 |
| `num_streams` | `1` | OpenVINO 内部执行 stream 数，必须为正整数。低延迟基线为 1；可测试 2，超过 2 前应先证明有收益。内部 stream 不等同于外部 `max_infer_num`。 |
| `enable_cpu_pinning` | 省略（自动） | `true`/`false`，控制推理线程绑核。只在目标操作系统和 CPU 上比较；省略时由 OpenVINO 决定。 |
| `scheduling_core_type` | 省略（自动） | `ANY_CORE`、`PCORE_ONLY` 或 `ECORE_ONLY`。仅对相应 CPU 拓扑有意义；减少可用核心可能降低吞吐。 |
| `infer_request_buffer_num` | `2` | OpenVINO infer request 资源池大小，必须为正整数。当前与双路外部检测匹配；通常不应小于 `max_infer_num`。 |

建议一次只改变一个资源维度，并记录硬件、模型、OpenVINO 版本、输入频率和系统负载。不要为每组参数复制完整 YAML。

## 3. 验证

启动日志必须出现实际生效值：

```text
OpenVINO effective inference_num_threads: ...
OpenVINO effective num_streams: ...
OpenVINO effective enable_cpu_pinning: ...
OpenVINO effective scheduling_core_type: ...
```

若实际值与预期不同，先检查 YAML 层级、字段拼写和 OpenVINO CPU 插件支持情况。

`cb5_perf` 每秒输出两行：

- 第一行：`tracked / detected / skipped`，以及 preprocess、infer、output-copy、postprocess 的 avg/P95；
- 第二行：传统 CV、Detector、Detector→Tracker queue、Tracker、total 的 avg/P95。

比较配置时必须保持相近输入和场景，重点观察：

1. camera Hz 与 detected Hz；
2. `skipped / tracked` 比例；
3. infer、Detector、total 的平均值和 P95；
4. 目标检测数量、识别结果和 Tracker 稳定性；
5. `pidstat -u -t -p "$(pgrep -n standard)" 1` 的线程 CPU 使用。

旧 `avg_latency` 可能受批量出队和时间归属影响，不能单独用于判断优劣。不能通过降低相机帧率或减少 Tracker 更新次数制造性能改善。

### 2026-10-28 Daedalus 对照结论

测试条件：Intel Core Ultra 7 255HX、CB5 `model-0624.onnx`、OpenVINO 2024.6、双路外部检测；只统计每秒 camera 为 57～60 的窗口。

| 配置 | 窗口数 | camera Hz | detect Hz | skip | infer 加权平均 | Detector 加权平均 | total 加权平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 stream | 10 | 58.50 | 42.70 | 27.67% | 33.28 ms | 35.32 ms | 35.34 ms |
| 2 streams | 33 | 58.55 | 37.94 | 35.33% | 39.89 ms | 41.81 ms | 42.40 ms |

本次端到端条件下 1 stream 的检测频率、skip 比例和平均耗时均更好，因此主配置保留 `num_streams: 1`。两组窗口数不同，且结论不推广到其他 CPU、模型或负载；历史独立 benchmark 中 2 streams 的结果不能替代端到端验证。

## 4. 恢复基线

出现检测频率下降、skip 增多、延迟增长、识别或 Tracker 回退时，停止进程，将主配置恢复为：

```yaml
rune_detector:
  max_infer_num: 2
  backend: openvino
  net_detector:
    openvino:
      model_path: ${ROOT_DIR}/model/model-0624.onnx
      device_name: "CPU"
      perf_mode: "LATENCY"
      priority: "HIGH"
      execution_mode: "PERFORMANCE"
      num_streams: 1
      infer_request_buffer_num: 2
    rune_infer:
      model_type: cb5
```

恢复时删除或注释 `inference_num_threads`、`enable_cpu_pinning` 和 `scheduling_core_type`，让 OpenVINO 自动选择，然后使用第 1 节命令重新启动。YAML 修改不需要重新编译。
