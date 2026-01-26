# gemini_336l_rtabmap_with336L.launch.py
# Compatible with ROS 2 Foxy + Orbbec Gemini 336L + RTAB-Map from source

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
            # 'enable_ir': 'true',
            # 'enable_imu': 'false',
        }.items()
    )

    # -------------------------- 4. RTAB-Map 通用参数 --------------------------
    parameters = [{
        'frame_id': 'camera_link',
        'subscribe_depth': True,
        'approx_sync': True,           # Gemini 336L 支持硬件同步
        'wait_imu_to_init': False,      # 未使用IMU
        'Odom/FeatureType': 'SURF',    # 可替换为 ORB/SIFT/AKAZE
        'Odom/MaxFeatures': 1000,      # 特征点数量
        'Odom/MinInliers': 10,         # 最小内点要求
        'Vis/EstimationType': '0',     # 视觉里程计模式
        'qos_image': 2,                # Best effort（匹配相机驱动默认QoS）
        'qos_camera_info': 2,
        'qos_scan': 2,
        'approx_sync_max_interval': 0.01,  # 同步最大时间差（10ms）
    }]

    # -------------------------- 5. 话题重映射（匹配相机输出） --------------------------
    remappings = [
        ('rgb/image', '/camera/color/image_raw'),
        ('rgb/camera_info', '/camera/color/camera_info'),
        ('depth/image', '/camera/depth/image_raw'),
    ]

    # -------------------------- 6. RTAB-Map 节点定义 --------------------------
    # 视觉里程计节点
    rgbd_odometry_node = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        output='screen',
        parameters=parameters,
        remappings=remappings,
        arguments=['--ros-args', '--log-level', 'info'],
    )

    # RTAB-Map SLAM 节点
    rtabmap_slam_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        output='screen',
        parameters=parameters,
        remappings=remappings,
        arguments=['-d'],  # 启动时删除旧数据库
    )

    # RTAB-Map 可视化节点
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        output='screen',
        parameters=parameters,
        remappings=remappings,
    )

    # -------------------------- 7. 组装所有启动项 --------------------------
    return LaunchDescription([
        # 先声明参数
        color_width_arg,
        color_height_arg,
        color_fps_arg,
        depth_width_arg,
        depth_height_arg,
        depth_fps_arg,
        # 再启动相机
        orbbec_camera_launch,
        # 最后启动RTAB-Map系列节点
        rgbd_odometry_node,
        rtabmap_slam_node,
        rtabmap_viz_node,
    ])