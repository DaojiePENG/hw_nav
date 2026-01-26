# camera_gemini_336l.launch.py
# 此 launch 文件用于启动奥比中光 Gemini 336L 相机节点

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    # -------------------------- 1. 声明可配置参数（可选，方便外部传参） --------------------------
    # 相机分辨率/帧率参数（默认值设为你需要的 1280x800@30Hz）
    color_width_arg = DeclareLaunchArgument(
        'color_width',
        default_value='1280',
        description='RGB image width'
    )
    color_height_arg = DeclareLaunchArgument(
        'color_height',
        default_value='800',
        description='RGB image height'
    )
    color_fps_arg = DeclareLaunchArgument(
        'color_fps',
        default_value='30',
        description='RGB image frame rate'
    )
    depth_width_arg = DeclareLaunchArgument(
        'depth_width',
        default_value='1280',
        description='Depth image width'
    )
    depth_height_arg = DeclareLaunchArgument(
        'depth_height',
        default_value='800',
        description='Depth image height'
    )
    depth_fps_arg = DeclareLaunchArgument(
        'depth_fps',
        default_value='30',
        description='Depth image frame rate'
    )
    enable_accel_arg = DeclareLaunchArgument(
        'enable_accel',
        default_value='true',
        description='Enable accelerometer'
    )
    enable_gyro_arg = DeclareLaunchArgument(
        'enable_gyro',
        default_value='true',
        description='Enable gyroscope'
    )
    # enable_ir_arg = DeclareLaunchArgument(
    #     'enable_ir',
    #     default_value='true',
    #     description='Enable infrared'
    # )
    # ir_width_arg = DeclareLaunchArgument(
    #     'ir_width',
    #     default_value='1280',
    #     description='Infrared image width'
    # )
    # ir_height_arg = DeclareLaunchArgument(
    #     'ir_height',
    #     default_value='800',
    #     description='Infrared image height'
    # )
    # ir_fps_arg = DeclareLaunchArgument(
    #     'ir_fps',
    #     default_value='30',
    #     description='Infrared image frame rate'
    # )

    # -------------------------- 2. 找到奥比中光相机的 launch 文件路径 --------------------------
    orbbec_camera_share_dir = get_package_share_directory('orbbec_camera')
    gemini_launch_file = os.path.join(
        orbbec_camera_share_dir,
        'launch',
        'gemini_330_series.launch.py'  # 对应你的相机启动文件
    )

    # -------------------------- 3. 导入相机 launch 文件并传递参数 --------------------------
    orbbec_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gemini_launch_file),
        launch_arguments={
            'color_width': LaunchConfiguration('color_width'),
            'color_height': LaunchConfiguration('color_height'),
            'color_fps': LaunchConfiguration('color_fps'),
            'depth_width': LaunchConfiguration('depth_width'),
            'depth_height': LaunchConfiguration('depth_height'),
            'depth_fps': LaunchConfiguration('depth_fps'),
            # 可选：添加其他相机参数（如启用红外、IMU等）
            'enable_accel': LaunchConfiguration('enable_accel'),
            'enable_gyro': LaunchConfiguration('enable_gyro'),
            # 'enable_ir': LaunchConfiguration('enable_ir'),
            # 'ir_width': LaunchConfiguration('ir_width'),
            # 'ir_height': LaunchConfiguration('ir_height'),
            # 'ir_fps': LaunchConfiguration('ir_fps'),
        }.items()
    )

    # -------------------------- 4. 组装所有启动项 --------------------------
    return LaunchDescription([
        # 先声明参数
        color_width_arg,
        color_height_arg,
        color_fps_arg,
        depth_width_arg,
        depth_height_arg,
        depth_fps_arg,
        enable_accel_arg,
        enable_gyro_arg,
        # enable_ir_arg,
        # ir_width_arg,
        # ir_height_arg,
        # ir_fps_arg,
        # 再启动相机
        orbbec_camera_launch,
    ])