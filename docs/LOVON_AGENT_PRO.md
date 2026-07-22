# LovonAgentPro：按自然语言特征跟随特定人物

> RK3588 的 RKNN 后端、异步 5 Hz 调度、模型转换和真板验收请直接阅读
> [LOVON_AGENT_PRO_RK3588.md](LOVON_AGENT_PRO_RK3588.md)。本文件前半部分的 SigLIP2/PyTorch 流程主要
> 面向 x86 CUDA/CPU；不能原样作为 RK3588 实时部署方案。

## 1. 交付结论

`LovonAgentPro` 已作为一条独立管线加入仓库。它不会修改或替换原有的
`lovon_control_loop.py`、`lovon/lovon_agent.py` 和 `pyproject.toml`，两套实现可以并排对比。

新管线解决的是原实现中的关键歧义：原 `LovonAgent` 在 YOLO 找到多个同类目标时执行
`max(detections, key=confidence)`，所以“人群里检测置信度最高的人”会被交给 l2mm；Pro 版先根据
语言描述选出特定人物并锁定其身份，之后才把该人物的一组 bbox 送入控制器。

默认模型和控制方式如下：

- 人物检测：Ultralytics YOLO11n；
- 短时跟踪：ByteTrack ID；
- 自由语言与人物裁剪匹配：Google SigLIP2 Base；
- 中文增强：原中文提示 + 确定性的常见穿搭英文提示，二者取更高匹配分数；
- 身份保持：track ID、bbox IoU、SigLIP2 外观向量和语言分数融合；
- 运动控制：默认使用可解释的 bbox 几何控制器；也可接回原 l2mm checkpoint；
- 安全策略：没有目标、匹配不明确、目标遮挡或推理异常时立即输出 `[0, 0, 0]`。

## 2. 为什么默认没有直接采用 PP-Human

PP-Human 是很好的安防属性方案，尤其适合上衣颜色、下衣颜色、帽子、背包、手提包等固定标签。
这里选择 SigLIP2 并不是因为 PP-Human 不可用，而是因为当前仓库已有 PyTorch + Ultralytics 技术栈，
再引入 PaddlePaddle、PaddleDetection 和第二套 CUDA/TensorRT 依赖，会显著增加 Jetson 上的部署冲突。

SigLIP2 的主要优势是：

- 支持中英文自由描述，不局限于预定义属性；
- 可以描述“最右边”“蓝白花纹”“金发”“黑色领带”等组合；
- 同一个图像向量同时用于语义匹配和遮挡后的外观关联；
- 官方 checkpoint 是 Apache-2.0，Transformers 可直接离线加载。

代价是零样本视觉语言分数需要现场标定，而且它不是严格的属性分类器。如果项目最终只允许固定属性、
要求输出每个属性的可审计概率，PP-Human 仍是更合适的替换后端。Pro 版把检测器和匹配器解耦为
`detector.detect(image)` 与 `matcher.score(image, detections, prompt)`，后续可在不改控制器的情况下接入
PP-Human。PP-Human 官方入口见 [PaddleDetection PP-Human](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.7/deploy/pphuman)。

SigLIP2 的多语言与零样本用法见 [官方模型卡](https://huggingface.co/google/siglip2-base-patch16-224)；
ByteTrack 的 `persist=True` 和持续 track ID 用法见
[Ultralytics 跟踪文档](https://docs.ultralytics.com/modes/track)。

> 许可证注意：SigLIP2 checkpoint 为 Apache-2.0；Ultralytics 软件和模型适用其当前许可证条款。
> 商业闭源部署前应自行确认 Ultralytics AGPL/企业许可证要求。

## 3. 数据流和设计

```text
用户指令
  │
  ├─ 中文/英文提示构造 ───────────────┐
  │                                  │
摄像头帧 → YOLO 人物框 → 逐人裁剪 → SigLIP2 图文分数 + 外观向量
                                      │
                     ByteTrack ID + IoU + 外观 + 语义
                                      │
                              稳定锁定一个人物
                                      │
                         仅输出该人物的归一化 bbox
                                      │
                        bbox 控制器或原 l2mm 控制器
                                      │
                             [Vx, Vy, Vz]
```

### 3.1 首次锁定

每个候选人的首次排序分数为：

```text
0.75 × text_match_score + 0.15 × spatial_score + 0.10 × detector_confidence
```

`spatial_score` 仅在指令明确包含“最左边/最右边/中间”等位置词时启用；否则它的 0.15 权重自动并回
`text_match_score`，等价于原来的 `0.90 × text + 0.10 × detector`。位置由完整画面中的 bbox 计算，弥补
逐人裁剪本身看不到“谁在最左边”的限制。

最高候选还必须同时满足：

- `text_match_score >= acquire_score_threshold`；
- 与第二名的差值不小于 `acquire_margin`。

因此 Pro 版不会在语言没有匹配到任何人时退回最高检测置信度的人。

### 3.2 连续身份关联

锁定后不再每帧重新按语言最高分选人，而使用：

```text
track_id_weight   × ID 是否相同
+ iou_weight      × 与上一帧 bbox 的 IoU
+ appearance_weight × 外观余弦相似度
+ semantic_weight   × 语言分数
```

这能降低两个人交叉时的跳人风险。目标暂时消失时，状态变为 `lost` 并立即停车；超过
`max_missed_frames` 后转回 `searching`，只有语义和保存的外观都支持时才重新锁定。

### 3.3 bbox 控制器

默认控制器复现用户描述的 l2mm 几何行为，且无需私有训练权重：

```text
横向误差 e = bbox_center_x - 0.5
Vz = clip(angular_sign × yaw_gain × e, ±max_yaw_speed)
```

- 偏差大于 `turn_in_place_error` 时先原地转向；
- bbox 宽度小于 `slow_down_bbox_width` 时按请求速度前进；
- 从 `slow_down_bbox_width` 到 `stop_bbox_width` 线性减速；
- bbox 宽度达到 `stop_bbox_width` 后输出零速；
- 指令中的 `0.3 m/s` 或 `0.3 米每秒` 会被解析，但始终受 `max_linear_speed` 限制。

这比黑盒 l2mm 更容易在不同相机视场角和底盘上标定。若必须复现实验中的原 l2mm，可使用第 10 节的
适配方式。

## 4. 新增文件

| 文件 | 用途 |
|---|---|
| `lovon/lovon_agent_pro.py` | Pro Agent、YOLO、SigLIP2、目标锁定、bbox/l2mm 控制适配 |
| `lovon_pro_control_loop.py` | 独立摄像头/视频/真机入口，默认 dry-run |
| `configs/lovon_agent_pro.yaml` | 在线模型 ID 默认配置 |
| `configs/lovon_agent_pro_offline.yaml` | 本地模型、断网运行配置 |
| `configs/lovon_agent_pro_l2mm.example.yaml` | 旧 l2mm 路径示例 |
| `scripts/create_lovon_pro_env.sh` | CPU、CUDA 12.4、Jetson 环境创建入口 |
| `scripts/download_lovon_pro_models.py` | 固定 revision 下载与 YOLO SHA-256 校验 |
| `scripts/smoke_test_lovon_pro.py` | mock 或单图片烟雾测试 |
| `scripts/evaluate_lovon_pro_image.py` | 多人图、多条指令评估和 JSON 报告 |
| `tests/test_lovon_agent_pro.py` | 不依赖模型权重的管线测试 |
| `models/lovon_pro/model_manifest.json` | 权重 URL、revision、hash 清单 |
| `lovon/rknn_backend.py` | RK3588 YOLO11/CLIP RKNNLite 后端和轻量 tracker |
| `lovon/realtime_runtime.py` | 最新帧异步感知与陈旧 bbox 停车 |
| `configs/lovon_agent_pro_rk3588.yaml` | RK3588 5 Hz 生产配置 |
| `docs/LOVON_AGENT_PRO_RK3588.md` | 从模型转换到真板 benchmark 的完整手册 |

## 5. 从 git clone 完整复现（x86 Linux）

### 5.1 前置条件

- Linux x86_64；
- Git；
- Miniconda、Anaconda 或 Miniforge；
- 首次下载至少约 2 GB 空间，建议预留 5 GB；
- CPU 可完成验证；实时运行建议 NVIDIA GPU；
- 摄像头测试需要 `/dev/video*` 访问权限。

以下命令均从仓库根目录执行。

```bash
git clone https://github.com/DaojiePENG/hw_nav.git
cd hw_nav
```

### 5.2 创建环境

CPU（可重复的最低基线）：

```bash
./scripts/create_lovon_pro_env.sh cpu
```

NVIDIA x86 GPU，驱动支持 CUDA 12.4 时：

```bash
./scripts/create_lovon_pro_env.sh cuda124
```

脚本创建 `lovon-pro`，安装 Python 3.10.18、PyTorch 2.5.1、TorchVision 0.20.1，并从
`requirements-lovon-pro-lock.txt` 安装完整解析后的固定依赖；`requirements-lovon-pro.txt` 则是便于阅读和
平台移植的直接依赖清单。也可以用声明式 CPU 环境：

```bash
conda env create -f environment-lovon-pro-cpu.yml
conda run -n lovon-pro python -m pip install --no-deps -e .
```

检查关键版本：

```bash
conda run -n lovon-pro python -c \
  "import torch, transformers, ultralytics; print(torch.__version__, transformers.__version__, ultralytics.__version__)"
```

预期核心版本为 `2.5.1`、`4.53.3`、`8.3.169`。

### 5.3 下载并校验模型

```bash
conda run -n lovon-pro python scripts/download_lovon_pro_models.py
conda run -n lovon-pro python scripts/download_lovon_pro_models.py --verify-only
```

固定资产是：

- `yolo11n.pt`：Ultralytics assets `v8.3.0`，SHA-256
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`；
- `google/siglip2-base-patch16-224`：revision
  `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`。

模型保存在 `models/lovon_pro/`，已被局部 `.gitignore` 排除，不会误提交 1.5 GB 权重。
完成后使用 `configs/lovon_agent_pro_offline.yaml` 即可断网运行。

### 5.4 自动测试

若系统安装过 ROS，ROS 的 pytest 插件可能通过 `PYTHONPATH` 泄漏到 Conda。以下命令显式禁用第三方
pytest 插件自动加载，任何电脑都可使用：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  conda run -n lovon-pro pytest -q tests/test_lovon_agent_pro.py
```

预期：`12 passed`。

无需任何模型的端到端 mock 烟雾测试：

```bash
conda run -n lovon-pro python scripts/smoke_test_lovon_pro.py --mock
```

实际模型单图测试：

```bash
conda run -n lovon-pro python scripts/smoke_test_lovon_pro.py \
  --config configs/lovon_agent_pro_offline.yaml \
  --image lovon/person.png \
  --instruction "跟随戴眼镜、穿白色衬衫、打黑色领带的男人"
```

## 6. `artifacts/multi-people.jpg` 真实验收

运行内置三条中文指令：

```bash
conda run -n lovon-pro python scripts/evaluate_lovon_pro_image.py
```

输出目录：`artifacts/lovon_pro_evaluation/`，包含三张标注图和 `report.json`。绿色框是最终目标，橙色框是
其他候选。内置三条指令还会按目标的横向区域自动断言，成功时最后显示
`PASS: all three default instructions selected the expected person region`；错误选择会让命令以非零状态退出。

本次在 CPU 环境和固定权重上的实测如下。ByteTrack ID 是运行期编号，在其他视频或版本中可能不同，
应核对 bbox 和绿色框，不应把 ID 数字写成业务规则。

| 指令 | 正确目标 | 实测 text score | bbox 中心 x | 控制结果 |
|---|---|---:|---:|---|
| 最右边穿橙色碎花连衣裙的金发女人 | 最右女性 | 0.973 | 0.801 | 先右转，`Vz < 0` |
| 最左边穿黑色 T 恤和蓝色牛仔裤的男人 | 最左男性 | 0.138 | 0.196 | 先左转，`Vz > 0` |
| 中间穿蓝白花纹上衣和黑色裤子的黑人男性 | 中间偏右男性 | 0.570 | 0.592 | 前进并小幅右转 |

第一轮仅使用中文提示时，第二条分数为 0.017，系统按安全阈值拒绝启动；加入常见穿搭的本地双语提示后
正确目标提升到 0.138，其他候选最高约 0.014，因此仍有清晰区分。这个过程说明为什么必须保留
`acquire_score_threshold` 和 `acquire_margin`，而不能无条件取图文分数最高者。

自定义一条或多条指令：

```bash
conda run -n lovon-pro python scripts/evaluate_lovon_pro_image.py \
  --instruction "跟随穿深色上衣和灰色短裙的长发女性" \
  --instruction "follow the rightmost blonde woman in an orange floral dress"
```

## 7. 摄像头和视频测试（默认不驱动电机）

普通 USB 摄像头：

```bash
conda run -n lovon-pro python lovon_pro_control_loop.py \
  --config configs/lovon_agent_pro_offline.yaml \
  --source 0 \
  --instruction "跟随戴红色帽子、背黑色双肩包的人"
```

视频文件：

```bash
conda run -n lovon-pro python lovon_pro_control_loop.py \
  --config configs/lovon_agent_pro_offline.yaml \
  --source test.mp4 \
  --instruction "跟随穿蓝色上衣的人" \
  --output artifacts/test_result.mp4
```

SSH/无显示器：增加 `--no-show`。退出窗口按 `q`，终端按 `Ctrl+C`。不传 `--drive` 时只打印
`[DRY-RUN]` 速度，不导入 `rosmaster_lib`，适合开发机测试。

`--interval-frames N` 会每 N 个原始帧执行一次推理。`max_missed_frames` 统计的是推理帧，因此实际遮挡
容忍时间约为：

```text
max_missed_frames × interval_frames / 摄像头 FPS
```

## 8. Rosmaster 真机运行

### 8.1 安装硬件库

硬件库与视觉依赖分开，开发机不必安装。真机执行：

```bash
conda run -n lovon-pro python -m pip install \
  "git+https://github.com/DaojiePENG/rosmaster_lib.git"
```

根据串口设备设置权限，例如：

```bash
sudo usermod -aG dialout "$USER"
```

重新登录后确认能访问 `/dev/ttyUSB*`。

### 8.2 上电前检查

1. 先架空驱动轮或让机器人位于空旷区域；
2. 准备物理急停或能立即断电；
3. 先用 dry-run 确认绿色框始终是目标；
4. 手推人物在画面左右，确认 `Vz` 符号正确；
5. 确认目标消失、遮挡和多人分数接近时速度为零；
6. 将 `max_linear_speed` 暂时降到 `0.1` 做第一次上地测试。

启用电机必须显式传 `--drive`：

```bash
conda run -n lovon-pro python lovon_pro_control_loop.py \
  --config configs/lovon_agent_pro_offline.yaml \
  --source 0 \
  --instruction "跟随穿红色上衣、背黑色包的人，速度为 0.2 米每秒" \
  --drive \
  --serial-port /dev/ttyUSB1
```

使用仓库已有的 Rosmaster 摄像头封装时，把 `--source 0` 换成 `--rosmaster-camera`。

无论正常退出、异常还是 `SIGINT/SIGTERM`，新控制循环都会在 `finally` 中发送零速并释放资源。不过软件
停车不能替代硬件急停。

## 9. 参数标定

所有参数位于 `configs/lovon_agent_pro.yaml`。

### 9.1 识别与锁定

- `detector.confidence`：低会增加误检，高会漏掉遮挡人物；建议从 `0.30` 开始；
- `selector.acquire_score_threshold`：无目标仍误锁时调高；正确目标经常被拒绝时根据评估报告调低；
- `selector.acquire_margin`：两人衣着相似时调高，让机器人等待更明确画面；
- `selector.min_appearance_similarity`：ID 经常切换时调高；遮挡后难以重连时小幅调低；
- `selector.max_missed_frames`：只改变保留身份的时间，不改变“丢失即停车”。

每个现场应收集至少三类短视频：目标单独出现、相似穿着多人交叉、目标完全离开。先运行视频 dry-run，
查看生成 JSON/日志后再改阈值。

### 9.2 距离与速度

1. 把目标站到期望停车距离；
2. 从状态日志记录 `object_whn[0]`；
3. 将它设为 `stop_bbox_width`，建议再留 10% 安全余量；
4. `slow_down_bbox_width` 设为停车宽度的约 60%～70%；
5. 从 `max_linear_speed: 0.10` 开始逐步提升；
6. 不要照搬另一台相机的 bbox 宽度：焦距、裁剪和安装角度都会改变该值。

### 9.3 转向符号

默认 `angular_sign: -1.0`：画面右侧目标产生负 `Vz`。如果底盘实际向反方向旋转，只将它改成 `1.0`，
不要同时更改 `yaw_gain`。符号确认后再调增益。

## 10. 复用原 l2mm

当前公开 Git 历史没有包含 `lovon/models_cxn_025/` 和 `lovon/yolo-models/`，原 `.gitignore` 也排除了
这些目录。因此一个干净 clone 无法仅靠原 `LovonAgent` 复现私有 l2mm。Pro 版默认 bbox 控制器正是为了
消除这个不可复现点。

如果另一台电脑已有原模型代码、tokenizer 和 checkpoint：

1. 按原目录结构放回 `lovon/models_cxn_025/`；
2. 复制 `configs/lovon_agent_pro_l2mm.example.yaml`；
3. 填写 `l2mm_tokenizer_path` 和 `l2mm_model_path`；
4. 将 `controller.backend` 设为 `l2mm`；
5. 先运行图片/视频 dry-run。

`L2MMMotionController` 会把自由语言选出的单一人物 bbox 写入与旧 Agent 相同的字段：
`predicted_object`、`confidence`、`object_xyn`、`object_whn`。为了避免 l2mm 在找不到特定人物时搜索到
其他人，Pro 适配器在 `target_visible=False` 时不会调用 checkpoint，而是直接返回零速。

## 11. Python API

```python
import cv2
from lovon.lovon_agent_pro import LovonAgentPro

agent = LovonAgentPro.from_config_file("configs/lovon_agent_pro_offline.yaml")
frame = cv2.imread("artifacts/multi-people.jpg")
state, motion_vector = agent.run(
    frame,
    user_instruction="跟随最右边穿橙色碎花连衣裙的金发女人",
)
annotated = agent.annotate(frame)
```

为方便替换原调用点，`run` 仍接受 `mission_instruction_0` 和 `mission_instruction_1`，但建议新代码明确使用
`user_instruction`。后续帧不改变指令时可以省略，它会保留当前目标；传入不同指令会立即清空旧锁定并
重新选择。

关键状态新增字段：

- `target_description`：去除“请跟随”等动作词后的描述；
- `target_match_score`：中文/英文提示中的最高 SigLIP2 概率；
- `target_track_id`：当前 ByteTrack ID；
- `candidate_count`：当前人物候选数；
- `selector_reason`：锁定、歧义或关联失败原因；
- `missed_frames`：连续丢失的推理帧数。

旧字段格式保持一致，因此控制层仍可读取 `[Vx, Vy, Vz]`。

## 12. Jetson 部署

Jetson 不能直接安装 x86 CUDA wheel；PyTorch wheel 必须与 JetPack/L4T 版本匹配。建议流程：

```bash
./scripts/create_lovon_pro_env.sh jetson
```

脚本会先创建 Python 3.10 环境并暂停。此时按照 NVIDIA 对应 JetPack 的官方说明安装
`torch`/`torchvision`，验证导入成功后回车，脚本继续安装其余依赖。若 JetPack 自带 Python 版本与 3.10
不匹配，应使用该 JetPack 官方 wheel 要求的 Python 环境，而不要强行改 wheel。

Jetson 优化顺序：

1. 先让 PyTorch FP16 版本正确运行；
2. 将 `detector.image_size` 从 640 降到 512 或 416；
3. 使用 `--interval-frames 2` 或更高；
4. 保持 SigLIP2 批量处理同帧所有人物；
5. 再考虑将 YOLO 导出 TensorRT；
6. 最后才考虑量化或替换更小的视觉语言模型，并重新采集阈值数据。

运行时关注内存、温度和降频。多人数量增加时，SigLIP2 的 crop batch 会线性增加显存和延迟。

## 13. 已知边界

- 自由语言匹配是零样本概率，不是身份认证或人脸识别；
- 完全相同穿着的人仍可能无法区分，此时应依靠位置描述或拒绝启动；
- 光照、背面、强遮挡、过小 bbox 会降低颜色和配饰可靠性；
- “最左边/最右边”适合首次锁定，锁定后身份优先，不会因为人物换位而换人；
- ByteTrack ID 不是跨视频永久身份；
- 不建议依赖种族、性别等敏感属性做业务决策，视觉语言模型可能存在偏差；优先使用衣着、配饰和位置；
- 当前控制只依据单目 bbox，没有真实深度和避障。真机必须结合底盘限速、急停和独立避障系统。

## 14. 验证旧实现未被修改

开发或合并前可执行：

```bash
git diff --exit-code -- lovon_control_loop.py lovon/lovon_agent.py pyproject.toml
```

没有输出且退出码为 0，表示三份原有参考实现仍保持仓库版本。
