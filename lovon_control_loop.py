#!conda hw_nav
# coding=utf-8

import os
import cv2
import time
import signal
import sys

# LOVON 相关
from lovon.lovon_agent import LovonAgent

# Rosmaster 控制
from rosmaster_lib import Rosmaster

# 使用你已有的 Rosmaster_Camera（如果可用）
try:
    from lovon.camera_rosmaster import Rosmaster_Camera
    USE_ROSMASTER_CAM = True
except ImportError:
    USE_ROSMASTER_CAM = False

# ================== 配置 ==================
MISSION_INSTRUCTION_0 = "run to the chair at speed of 0.4 m/s"
MISSION_INSTRUCTION_1 = "run to the chair at speed of 0.4 m/s"
LOVON_INTERVAL_FRAMES = 5  # 每5帧处理一次，避免过载
SHOW_VIDEO = True  # 是否显示实时画面（调试用）
USE_CAMERA_INDEX = 0  # 如果不用 Rosmaster_Camera，用 OpenCV 默认摄像头

# ================== 初始化 ==================
print("Initializing LOVON Agent...")
lovon_agent = LovonAgent(wh_scale_factor=1.2, velocity_scale=1.2)  # 可根据需要调整缩放因子，避免机器人碰撞

print("Initializing Rosmaster bot...")
bot = Rosmaster()
bot.create_receive_threading()
bot.set_auto_report_state(enable=True, forever=False)

# 摄像头初始化
if USE_ROSMASTER_CAM:
    print("Using Rosmaster_Camera...")
    camera = Rosmaster_Camera(debug=False)
else:
    print(f"Using OpenCV camera index {USE_CAMERA_INDEX}...")
    cap = cv2.VideoCapture(USE_CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Cannot open camera!")
        sys.exit(1)

# 全局控制标志
running = True

def signal_handler(sig, frame):
    global running
    print("\nShutting down gracefully...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# ================== 控制函数 ==================
def car_motion(speed_x, speed_y, speed_z):
    speed_x = speed_x
    speed_y = speed_y
    speed_z = speed_z
    bot.set_car_motion(speed_x, speed_y, speed_z)
    print(f"[Control] speed_x={speed_x:.2f} m/s, speed_y={speed_y:.2f} m/s, speed_z={speed_z:.2f} rad/s")

# ================== 主循环 ==================
frame_count = 0

try:
    while running:
        # 获取图像
        if USE_ROSMASTER_CAM:
            success, frame = camera.get_frame()
            if not success:
                print("Camera failed, reconnecting...")
                camera.reconnect()
                time.sleep(0.5)
                continue
        else:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from OpenCV camera")
                break

        # 每 N 帧处理一次
        if frame_count % LOVON_INTERVAL_FRAMES == 0:
            # LOVON 推理
            try:
                state, motion_vector = lovon_agent.run(
                    image=frame,
                    mission_instruction_0=MISSION_INSTRUCTION_0,
                    mission_instruction_1=MISSION_INSTRUCTION_1
                )
                print(f"[LOVON] State: {state}, Motion: {motion_vector}")

                if motion_vector is not None and len(motion_vector) == 3:
                    V_x, V_y, V_z = motion_vector
                    car_motion(V_x, V_y, V_z)
                else:
                    print("[LOVON] Invalid motion vector, stopping.")
                    bot.set_car_motion(0, 0, 0)

            except Exception as e:
                print(f"[ERROR] LOVON inference failed: {e}")
                bot.set_car_motion(0, 0, 0)

        # 显示画面（可选）
        if SHOW_VIDEO:
            cv2.imshow("LOVON Control - Press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1
        time.sleep(0.001)  # 防止 CPU 占用过高

except KeyboardInterrupt:
    pass

# ================== 清理资源 ==================
print("Stopping robot...")
bot.set_car_motion(0, 0, 0)
time.sleep(0.1)

if USE_ROSMASTER_CAM:
    camera.clear()
else:
    cap.release()

cv2.destroyAllWindows()
print("Program exited.")