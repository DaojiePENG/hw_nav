# LovonAgentPro 在 RK3588 上的部署、复现与 5 Hz 验收

## 1. 结论和适用范围

本仓库现在包含一条不依赖板端 PyTorch 的 RK3588 路径：

- 人体检测：Rockchip 优化版 YOLO11n，INT8 RKNN；
- 帧间跟踪：CPU 上的保守 IoU Track ID；
- 语言匹配：OpenAI CLIP ViT-B/32 的图像、文本双 RKNN，FP16；
- 调度：首次获取、目标 ID 丢失和周期复核时运行 CLIP；锁定后只运行 YOLO、跟踪和 bbox 控制；
- 控制：异步读取最新结果，默认 5 Hz；bbox 数据超过 400 ms、推理异常或目标丢失时强制零速。

这套设计的验收目标是“目标锁定后的控制输出稳定达到 5 Hz”，不是“对画面内每个人每 200 ms
重新做一次 CLIP”。Rockchip v2.3.2 的官方纯模型数据中，RK3588 单核 YOLO11n INT8 为 60.0 FPS，
CLIP 图像编码器 FP16 为 6.5 FPS。后者每个人约 154 ms，四个人仅图像编码就约 615 ms，所以必须移出
控制关键路径。这些数字不含预处理和后处理，来源见
[RKNN Model Zoo v2.3.2](https://github.com/airockchip/rknn_model_zoo/tree/v2.3.2)。

当前开发机不是 RK3588，仓库内已完成无 NPU 的接口、张量解码、调度和安全测试；最终 5 Hz 结论必须在
目标板、目标 BSP、摄像头和散热条件下执行第 8 节 benchmark，不能用 x86 模拟结果替代。

## 2. 运行架构

```text
摄像头 20~30 FPS
      │ 只保留最新帧，旧待处理帧直接丢弃
      ▼
YOLO11n INT8 / RKNN NPU ──→ IoU Track ID / CPU
      │
      ├─ 新指令、尚未锁定、目标 ID 消失
      │       └─→ CLIP 图像 RKNN × 全部候选人数
      │             + 每条指令只运行一次并缓存的 CLIP 文本 RKNN
      │
      ├─ 2 秒周期复核：CLIP 只处理当前 Track ID 的一个裁剪
      │       └─ 语义/外观校验失败：本帧停车，下一帧对全部候选重获
      │
      └─ 其余已锁定帧：不运行 CLIP
                    │
                    ▼
             最新目标 bbox + 时间戳
                    │
          5 Hz 控制线程读取最新结果
                    │
          ┌─────────┴─────────┐
      age <= 400 ms        过期/异常/丢失
          │                   │
      bbox 控制器          [0, 0, 0]
```

重要不变量：

1. 感知队列长度永远为 1，不积累历史帧延迟；
2. 锁定后的正常帧 `state.matcher_ran == false`；
3. 控制使用采集时间而不是推理完成时间计算 bbox 年龄；
4. 新语言指令立即清空旧身份、停止发布旧指令的在途结果，并强制用新文本重新匹配；
5. 周期复核不通过时不污染身份模板，立即停车并强制重新匹配全部候选；
6. RKNN 模型缺失、Runtime 导入失败或输出布局不符时直接报错并停车，不退回 CPU SigLIP2。

## 3. 新增实现

| 文件 | 作用 |
|---|---|
| `lovon/rknn_backend.py` | RKNNLite 封装、YOLO11 DFL/NMS、IoU tracker、CLIP tokenizer/图像匹配 |
| `lovon/realtime_runtime.py` | 最新帧异步 worker、丢帧策略、陈旧结果停车 |
| `configs/lovon_agent_pro_rk3588.yaml` | 5 Hz 板端配置 |
| `scripts/create_lovon_rk3588_env.sh` | 固定 RKNN Toolkit 2.3.2 的转换端/板端环境 |
| `scripts/download_lovon_rk3588_models.py` | 下载并校验三份官方 ONNX 和固定 revision tokenizer |
| `scripts/convert_lovon_rk3588_models.py` | YOLO INT8、CLIP FP16 转换并生成哈希报告 |
| `scripts/prepare_lovon_rk3588_models.sh` | 创建环境、下载、校验、转换的一键入口 |
| `scripts/verify_lovon_rk3588_bundle.py` | 复制后按本次转换报告校验三个 RKNN 和 tokenizer |
| `scripts/benchmark_lovon_rk3588.py` | 真板首次获取、锁定路径 p50/p95 和异步控制节拍验收 |
| `tests/test_lovon_rk3588.py` | 不需要 NPU 的后端与实时安全测试 |

RK3588 产品规格为 6 TOPS NPU，支持 INT8/FP16 等精度，见
[Rockchip RK3588 官方页面](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html)。

## 4. 固定版本和模型资产

转换工具、模型示例和 Runtime 必须使用同一代 RKNPU SDK。这里固定：

- RKNN Toolkit2 / Lite2：`v2.3.2`，commit `42aa1d426c0a9e0869b6374edba009f7208a1926`；
- RKNN Model Zoo：`v2.3.2`，commit `bad6c7334531becaf90a561988519b7bec34d0ab`；
- Python 3.10、`setuptools<81`（RKNN 仍使用 `pkg_resources`）、ONNX `1.16.1`（保留 RKNN 使用的
  `onnx.mapping` API）；
- CLIP tokenizer：revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`。

下载脚本依据下表做文件大小和 SHA-256 双重检查：

| 资产 | 大小 | SHA-256 |
|---|---:|---|
| `yolo11n.onnx` | 10,527,859 | `62a3751662e06c678debb54b113c211d918f245ea6d3aea0b09fc418b8fc7705` |
| `clip_images.onnx` | 351,779,834 | `981517696b79c0dcd8c3b03564369213ef891f0cc9f7692e0cfa89ab58d471ff` |
| `clip_text.onnx` | 254,185,587 | `9cd686c57d874b4f7ec8e539fecc9ac2367c4252be18d662cd7bd23e872fddee` |

清单的机器可读版本是 `models/lovon_pro_rk3588/model_manifest.json`。RKNN 文件由 Toolkit、量化图片和
目标平台共同生成，所以转换脚本另写出 `conversion_report.json`，记录输入、输出哈希和工具版本。
仓库默认测试图的转换已经在 x86_64 上实际执行，并确认三个 `.rknn` 均可由 Toolkit 重新载入。重复转换
表明 RKNN 序列化不是字节级确定性的，因此输出 SHA-256 用于追踪某一次构建和传输校验，不能作为跨电脑
相等的验收标准；跨电脑应核对输入哈希、量化集哈希、工具链版本、`load_rknn: ok` 和板端推理结果。
使用正式量化集后 YOLO INT8 输出变化也是预期行为。

## 5. x86_64 电脑生成 RKNN 模型

### 5.1 前置条件

- Ubuntu 20.04/22.04 x86_64；
- Git、Miniforge/Miniconda；
- 至少 5 GB 可用空间；
- 可访问 GitHub、Hugging Face 和 Rockchip 模型下载地址。

从干净目录开始：

```bash
git clone https://github.com/DaojiePENG/hw_nav.git
cd hw_nav
git rev-parse HEAD
```

保存最后一条输出；板端必须 checkout 同一个 commit。

### 5.2 一键创建转换环境、下载和转换

```bash
chmod +x scripts/create_lovon_rk3588_env.sh \
  scripts/prepare_lovon_rk3588_models.sh
./scripts/prepare_lovon_rk3588_models.sh
```

脚本会创建 Conda 环境 `lovon-rknn-convert`，只从 RKNN Toolkit2 `v2.3.2` 安装 Python 3.10 的 x86
wheel，然后下载约 617 MB ONNX。输出应包含：

```text
models/lovon_pro_rk3588/yolo11n_i8.rknn
models/lovon_pro_rk3588/clip_images_fp16.rknn
models/lovon_pro_rk3588/clip_text_fp16.rknn
models/lovon_pro_rk3588/clip-tokenizer/
models/lovon_pro_rk3588/conversion_report.json
```

只验证下载资产：

```bash
conda run -n lovon-rknn-convert python \
  scripts/download_lovon_rk3588_models.py --verify-only
```

### 5.3 INT8 量化图片

缺省只使用仓库的 `artifacts/multi-people.jpg`，用于验证转换链路，不足以代表真实场景。正式部署应从
目标相机采集至少 100 张不重复图片，覆盖室内外、明暗、远近、遮挡和多人；图片不需要人工标注。

```bash
conda run -n lovon-rknn-convert python \
  scripts/convert_lovon_rk3588_models.py --only yolo \
  --calibration-image calibration/001.jpg \
  --calibration-image calibration/002.jpg \
  --calibration-image calibration/003.jpg
```

可重复增加 `--calibration-image`。量化集变化会使 RKNN 哈希变化，这是预期行为；保留对应的
`conversion_report.json` 和量化图片清单。

## 6. RK3588 板端环境

### 6.1 BSP 和驱动门槛

要求 64 位 Linux (`uname -m` 输出 `aarch64`)。Model Zoo v2.3.2 要求 RKNPU2 SDK/驱动不低于 2.3.2；
不同厂商开发板更新 BSP 的方法不同，应使用板卡厂商提供的完整镜像/驱动包，不能只从其他系统复制一个
`librknnrt.so`。

```bash
uname -m
sudo cat /sys/kernel/debug/rknpu/version
```

若第二条路径不存在，先确认 debugfs 是否挂载，并查阅板卡 BSP 文档。工具与驱动版本不匹配可能表现为
初始化失败、错误输出或性能异常。官方版本关系见
[RKNN Model Zoo 环境依赖](https://github.com/airockchip/rknn_model_zoo/tree/v2.3.2#environment-dependencies)。

建议至少 8 GB RAM，并确保 NPU/CPU 有散热片和风扇；验收时记录是否发生降频。

### 6.2 clone 相同代码并创建环境

```bash
git clone https://github.com/DaojiePENG/hw_nav.git
cd hw_nav
git checkout <第5节记录的commit>
chmod +x scripts/create_lovon_rk3588_env.sh
./scripts/create_lovon_rk3588_env.sh board
```

脚本创建 `lovon-rk3588`，安装固定的 RKNNLite 2.3.2 aarch64 wheel、OpenCV headless、NumPy、
Transformers tokenizer 和本仓库。板端不会安装 Torch、Ultralytics 或 SigLIP2。

### 6.3 复制模型

从转换电脑复制生成结果到板端相同目录。最简单且最不容易漏文件的方式：

```bash
scp -r models/lovon_pro_rk3588 \
  <user>@<rk3588-ip>:<hw_nav绝对路径>/models/
```

板端确认：

```bash
ls -lh models/lovon_pro_rk3588/*.rknn
conda run -n lovon-rk3588 python -c \
  "from rknnlite.api import RKNNLite; print('RKNNLite import OK')"
conda run -n lovon-rk3588 python scripts/verify_lovon_rk3588_bundle.py
```

ONNX 只用于 x86 转换。存储空间紧张时，板端可以删除三份 `.onnx`，但必须保留三个 `.rknn`、
`clip-tokenizer/` 和 `conversion_report.json`。

## 7. 自动测试和静态图片烟雾测试

开发机或板端均可运行不加载模型的测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n lovon-pro pytest -q \
  tests/test_lovon_agent_pro.py tests/test_lovon_rk3588.py
```

板端环境名替换为 `lovon-rk3588`。测试覆盖：

- Rockchip YOLO11 六输出布局的 DFL 解码和 person NMS；
- IoU Track ID 连续性和不复用；
- CLIP 只采用英文提示、文本 embedding 缓存；
- 事件调度在连续 5 Hz 帧中不调用 CLIP；
- 新指令强制匹配；
- 感知繁忙时丢弃旧待处理帧；
- 400 ms 内允许控制，超过阈值立即停车。

板端加载真实 NPU 模型：

```bash
conda run -n lovon-rk3588 python scripts/smoke_test_lovon_pro.py \
  --config configs/lovon_agent_pro_rk3588.yaml \
  --image artifacts/multi-people.jpg \
  --instruction "跟随长头发、穿褐色上衣的女生" \
  --output artifacts/rk3588_smoke.jpg
```

该中文指令会确定性转换为 `woman long hair wearing brown shirt`，文本向量只计算一次。检查输出绿框，
不要只看命令是否为零退出。

## 8. 真板 5 Hz 性能验收

固定最高性能模式和频率的方法由板卡厂商 BSP 决定；先开启主动散热，再执行：

```bash
conda run -n lovon-rk3588 python scripts/benchmark_lovon_rk3588.py \
  --config configs/lovon_agent_pro_rk3588.yaml \
  --image artifacts/multi-people.jpg \
  --instruction "跟随最右边穿橙色碎花连衣裙的金发女人" \
  --iterations 50 \
  --required-control-hz 5 \
  --control-duration 5 \
  --camera-hz 20
```

报告保存到 `artifacts/rk3588_benchmark.json`。通过条件同时为：

```text
locked_path.p95_ms <= 200.0
locked_matcher_calls == 0
async_control.interval.p95_ms <= 250.0
async_control.measured_hz >= 4.75
首次锁定后的 tracking 控制 tick 比例 >= 95%
```

`acquisition_ms` 单独报告，不纳入 200 ms 门槛。多人首次获取约需“候选人数 × CLIP 图像编码时间”，
因此可能为 0.5~1 秒；搜索期间控制输出为零。异步验收会以 20 Hz 提交静态图像，确认旧帧被丢弃、
控制线程仍以 5 Hz 独立发布，并按源帧时间戳拒绝陈旧结果；该阶段会恢复配置中的 2 秒 CLIP 周期复核，
不是只测关闭 CLIP 后的理想路径。不能只报告平均 FPS，必须看 p95 和温度稳定后的持续结果。

建议再用现场视频分别测试：目标单人、四人交叉、相似穿着、完全遮挡。每种至少运行 10 分钟，记录：

- locked path p50/p95/max；
- 首次锁定和遮挡重获时间；
- `stale_target` 次数；
- 误切换目标次数；
- NPU/CPU 温度和降频；
- 目标消失后实际停车时间。

## 9. 摄像头 dry-run 和真车

先不接电机：

```bash
conda run -n lovon-rk3588 python lovon_pro_control_loop.py \
  --config configs/lovon_agent_pro_rk3588.yaml \
  --source 0 \
  --instruction "跟随长头发、穿褐色上衣的女生" \
  --no-show
```

`runtime.async_perception: true` 会自动启用异步模式；也可显式传 `--runtime-mode async`。日志应显示控制
原因 `tracking`，锁定后的 matcher 原因为 `tracking_cache`。若看到 `stale_target`，先检查相机阻塞、
NPU Runtime、温度和是否错误地频繁运行 CLIP，不要直接放宽超时。

上车前先把 `max_linear_speed` 降到 `0.10`，架空车轮确认角速度方向、目标消失停车和物理急停。确认后：

```bash
conda run -n lovon-rk3588 python -m pip install \
  "git+https://github.com/DaojiePENG/rosmaster_lib.git"

conda run -n lovon-rk3588 python lovon_pro_control_loop.py \
  --config configs/lovon_agent_pro_rk3588.yaml \
  --source 0 \
  --instruction "跟随长头发、穿褐色上衣的女生，速度为0.15米每秒" \
  --drive --serial-port /dev/ttyUSB1 --no-show
```

软件停车不能替代硬件急停和独立避障。

## 10. 配置说明

### 10.1 实时调度

| 参数 | 缺省 | 含义 |
|---|---:|---|
| `runtime.control_hz` | 5.0 | 向底盘写入最新速度的频率 |
| `runtime.max_target_age_sec` | 0.40 | 源图像时间戳超过此值立即停车 |
| `scheduling.search_interval_sec` | 0.50 | 搜索/重获时两次 CLIP 的最小间隔 |
| `scheduling.refresh_interval_sec` | 2.00 | 已锁定身份的周期外观复核间隔 |
| `detector.tracker_max_missed` | 15 | IoU tracker 保留未匹配 ID 的检测帧数 |
| `detector/matcher.core_mask` | `auto` | 交给 RKNNLite 的 `NPU_CORE_AUTO` 分配 NPU 核 |

`refresh_interval_sec` 越小越容易及时发现 ID 错误，但 NPU 占用和控制陈旧风险越高。不要设为 `0` 来表示
逐帧刷新：在实现中 `0` 表示关闭周期刷新，仍会在新指令或目标 ID 消失时匹配。
`core_mask` 也支持 `core0`、`core1`、`core2`、`core0_1_2` 和 `all`；缺少针对目标 BSP 的对照 benchmark
时应保留 `auto`，不要仅凭核心数量假定组合模式一定更快。

### 10.2 中文指令范围

Rockchip 官方 CLIP checkpoint 是 `openai/clip-vit-base-patch32`，模型卡说明其使用英文文本，见
[CLIP ViT-B/32 model card](https://huggingface.co/openai/clip-vit-base-patch32)。RK3588 配置因此先用确定性词典
将常见中文转为英文，当前覆盖：

- 黑、白、红、蓝、黄、灰、绿、橙、紫、粉、棕、褐、咖啡色；
- 上衣、衬衫、T 恤、外套、裤子、牛仔裤、裙子、连衣裙；
- 长/短头发、帽子、眼镜、太阳镜、背包、手提包；
- 男/女和最左、最右、中间位置。

词典无法完整转换时会拒绝启动，而不是把中文直接交给英文 CLIP 后误跟。需要开放域中文时有三种选择：

1. 扩充 `translate_common_zh_attributes()` 并为每个新属性增加现场测试；
2. 在指令入口增加离线中文到结构化英文属性解析器，结果仍缓存；
3. 换成经 RKNN 验证的多语言轻量模型。当前 SigLIP2 没有官方 RKNN 验证，不能直接宣称可用。

### 10.3 CLIP 阈值

RKNN CLIP matcher 输出归一化余弦相似度 `(cosine + 1) / 2`，不是 SigLIP2 的 sigmoid 概率。配置中的
`acquire_score_threshold: 0.60` 和 `acquire_margin: 0.015` 是偏保守的起点。现场应为每种语言属性收集正负样本，
优先保证无匹配和多人接近时拒绝启动，再决定是否降低阈值。

## 11. 达不到 5 Hz 时的排查顺序

1. 确认日志中锁定帧为 `matcher_reason=tracking_cache`；若仍是 `periodic_refresh`，检查 Track ID 是否抖动；
2. 确认正在使用三个 `.rknn`，没有启动原 `configs/lovon_agent_pro_offline.yaml`；
3. 检查 `conversion_report.json` 的 Toolkit 为 2.3.2，板端 Runtime/驱动不低于 2.3.2；
4. 单独运行 benchmark，区分首次 CLIP 慢和 locked path 慢；
5. 检查相机读取是否阻塞、图像是否发生多次颜色转换/复制；
6. 检查温度和 NPU/CPU 降频；
7. 最后才尝试把 YOLO 输入从 640 改为 512，并重新转换模型、量化和评估检测召回率。

如果锁定路径 p95 通过但底盘控制仍不稳定，应检查串口写入和摄像头线程，而不是继续缩小视觉模型。
