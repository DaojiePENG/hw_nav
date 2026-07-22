#!/usr/bin/env python3
"""Standalone camera/robot loop for :class:`lovon.lovon_agent_pro.LovonAgentPro`.

The robot is dry-run by default.  Passing ``--drive`` is an explicit safety
acknowledgement that enables Rosmaster motor commands.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2

from lovon.lovon_agent_pro import LovonAgentPro
from lovon.realtime_runtime import AsyncLovonRuntime, ControlTick, FixedRateControlPublisher


LOGGER = logging.getLogger("lovon_pro_control")
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "lovon_agent_pro.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据中英文人物特征指令锁定并跟随特定的人",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--instruction", help="例如：跟随穿红色上衣、背黑色双肩包的人")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Pro YAML 配置")
    parser.add_argument("--source", default="0", help="摄像头编号、视频文件或 RTSP URL")
    parser.add_argument("--rosmaster-camera", action="store_true", help="使用 lovon.camera_rosmaster")
    parser.add_argument("--drive", action="store_true", help="实际向 Rosmaster 下发速度；缺省仅打印")
    parser.add_argument("--serial-port", help="Rosmaster 串口，例如 /dev/ttyUSB1；留空使用库默认值")
    parser.add_argument("--interval-frames", type=int, default=1, help="每隔多少帧执行一次完整推理")
    parser.add_argument(
        "--runtime-mode",
        choices=("auto", "sync", "async"),
        default="auto",
        help="auto 按配置选择；RK3588 应使用 async",
    )
    parser.add_argument("--control-hz", type=float, help="异步模式控制指令频率，留空读取配置")
    parser.add_argument("--max-target-age", type=float, help="bbox 超过该秒数未更新就停车，留空读取配置")
    parser.add_argument("--no-show", action="store_true", help="无 GUI/SSH 环境中禁用窗口")
    parser.add_argument("--output", type=Path, help="可选：保存带标注的视频")
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示持续运行，正数用于烟雾测试")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def _opencv_source(raw: str) -> Any:
    return int(raw) if raw.isdecimal() else raw


class FrameSource:
    def read(self) -> Tuple[bool, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class OpenCVSource(FrameSource):
    def __init__(self, source: str) -> None:
        self.capture = cv2.VideoCapture(_opencv_source(source))
        if not self.capture.isOpened():
            raise RuntimeError(f"无法打开视频源：{source}")

    def read(self) -> Tuple[bool, Any]:
        return self.capture.read()

    def close(self) -> None:
        self.capture.release()


class RosmasterCameraSource(FrameSource):
    def __init__(self) -> None:
        try:
            from lovon.camera_rosmaster import Rosmaster_Camera
        except ImportError as exc:
            raise RuntimeError("无法导入 lovon.camera_rosmaster.Rosmaster_Camera") from exc
        self.camera = Rosmaster_Camera(debug=False)

    def read(self) -> Tuple[bool, Any]:
        success, frame = self.camera.get_frame()
        if not success:
            LOGGER.warning("Rosmaster 摄像头读取失败，尝试重连")
            self.camera.reconnect()
        return success, frame

    def close(self) -> None:
        self.camera.clear()


class MotionSink:
    def command(self, vector: Any) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class DryRunMotionSink(MotionSink):
    def command(self, vector: Any) -> None:
        LOGGER.info("[DRY-RUN] Vx=%+.3f Vy=%+.3f Vz=%+.3f", *vector)

    def stop(self) -> None:
        LOGGER.info("[DRY-RUN] STOP")


class RosmasterMotionSink(MotionSink):
    def __init__(self, serial_port: Optional[str]) -> None:
        try:
            from rosmaster_lib import Rosmaster
        except ImportError as exc:
            raise RuntimeError("--drive 需要 rosmaster_lib；请执行安装手册中的硬件依赖步骤") from exc
        self.bot = Rosmaster(com=serial_port) if serial_port else Rosmaster()
        self.bot.create_receive_threading()
        self.bot.set_auto_report_state(enable=True, forever=False)
        self.stop()

    def command(self, vector: Any) -> None:
        self.bot.set_car_motion(float(vector[0]), float(vector[1]), float(vector[2]))

    def stop(self) -> None:
        self.bot.set_car_motion(0.0, 0.0, 0.0)


def _open_writer(path: Path, frame: Any, fps: float = 20.0) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频：{path}")
    return writer


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.interval_frames <= 0:
        raise ValueError("--interval-frames 必须大于 0")
    instruction = args.instruction
    if not instruction:
        if not sys.stdin.isatty():
            raise ValueError("非交互模式必须传入 --instruction")
        instruction = input("请输入要跟随的人物特征：").strip()
    if not instruction:
        raise ValueError("人物特征指令不能为空")

    LOGGER.info("目标指令：%s", instruction)
    LOGGER.info("加载配置：%s", args.config)
    agent = LovonAgentPro.from_config_file(args.config)
    runtime_config = agent.config["runtime"]
    async_mode = (
        bool(runtime_config.get("async_perception", False))
        if args.runtime_mode == "auto"
        else args.runtime_mode == "async"
    )
    control_hz = float(
        args.control_hz if args.control_hz is not None else runtime_config.get("control_hz", 5.0)
    )
    max_target_age = float(
        args.max_target_age
        if args.max_target_age is not None
        else runtime_config.get("max_target_age_sec", 0.40)
    )
    if control_hz <= 0.0:
        raise ValueError("control_hz 必须大于 0")
    if max_target_age <= 0.0:
        raise ValueError("max_target_age 必须大于 0")
    frame_source: FrameSource = RosmasterCameraSource() if args.rosmaster_camera else OpenCVSource(args.source)
    motion_sink: MotionSink = RosmasterMotionSink(args.serial_port) if args.drive else DryRunMotionSink()
    LOGGER.warning("电机输出：%s", "已启用" if args.drive else "关闭（dry-run）")

    running = True

    def request_shutdown(_signum: int, _frame: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    frame_index = 0
    processed_frames = 0
    last_submitted_sequence: Optional[int] = None
    writer: Optional[cv2.VideoWriter] = None
    annotated = None
    async_runtime: Optional[AsyncLovonRuntime] = None
    control_publisher: Optional[FixedRateControlPublisher] = None
    if async_mode:
        async_runtime = AsyncLovonRuntime(
            agent,
            max_target_age_sec=max_target_age,
            annotate=bool(args.output or not args.no_show),
        )

        def log_control_tick(tick: ControlTick) -> None:
            state = tick.snapshot.state or {}
            LOGGER.info(
                "control=%s search=%s selector=%s id=%s matcher=%s motion=%s infer=%.1fms age=%s",
                tick.reason,
                state.get("search_state_in", "not_ready"),
                state.get("selector_reason", "not_ready"),
                state.get("target_track_id"),
                state.get("matcher_reason", "not_ready"),
                [round(value, 3) for value in tick.motion],
                tick.snapshot.inference_ms,
                (
                    "n/a"
                    if tick.snapshot.frame_timestamp is None
                    else f"{max(0.0, tick.timestamp - tick.snapshot.frame_timestamp) * 1000.0:.1f}ms"
                ),
            )

        control_publisher = FixedRateControlPublisher(
            async_runtime,
            motion_sink.command,
            frequency_hz=control_hz,
            on_tick=log_control_tick,
        )
        control_publisher.start()
        LOGGER.info("异步感知已启用：control_hz=%.2f stale_timeout=%.3fs", control_hz, max_target_age)

    try:
        while running:
            success, frame = frame_source.read()
            if not success:
                LOGGER.info("视频源结束或暂时不可用")
                break
            if frame_index % args.interval_frames == 0:
                if async_runtime is not None:
                    last_submitted_sequence = async_runtime.submit_frame(frame, instruction)
                else:
                    started = time.perf_counter()
                    try:
                        state, motion = agent.run(frame, user_instruction=instruction)
                        motion_sink.command(motion)
                        annotated = agent.annotate(frame)
                        elapsed_ms = (time.perf_counter() - started) * 1000.0
                        LOGGER.info(
                            "state=%s search=%s reason=%s id=%s match=%.3f candidates=%d motion=%s %.1fms",
                            state["mission_state_in"],
                            state["search_state_in"],
                            state["selector_reason"],
                            state["target_track_id"],
                            state["target_match_score"],
                            state["candidate_count"],
                            [round(value, 3) for value in motion],
                            elapsed_ms,
                        )
                    except Exception:
                        motion_sink.stop()
                        LOGGER.exception("推理失败，已发送停车指令")
                        if args.drive:
                            raise
                        annotated = frame.copy()
                processed_frames += 1
            elif annotated is None:
                annotated = frame

            if async_runtime is not None:
                if control_publisher is not None:
                    control_publisher.raise_if_failed()
                snapshot = async_runtime.get_snapshot()
                if snapshot.annotated is not None:
                    annotated = snapshot.annotated
                elif annotated is None:
                    annotated = frame

            if args.output:
                if writer is None:
                    writer = _open_writer(args.output, annotated)
                writer.write(annotated)
            if not args.no_show:
                cv2.imshow("LovonAgentPro | q=quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break
        # A finite video or --max-frames smoke test must not clear its final
        # pending frame before the worker has produced a result.
        if async_runtime is not None and last_submitted_sequence is not None:
            try:
                async_runtime.wait_until_processed(last_submitted_sequence, timeout=30.0)
            except TimeoutError:
                LOGGER.warning("等待最后一帧异步推理超时；退出时保持停车")
            if control_publisher is not None:
                control_publisher.raise_if_failed()
    finally:
        if control_publisher is not None:
            control_publisher.close()
        motion_sink.stop()
        if async_runtime is not None:
            async_runtime.close()
        frame_source.close()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        LOGGER.info("已停车并释放资源")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
