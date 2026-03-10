from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    use_sim_time = LaunchConfiguration('use_sim_time')
    qos = LaunchConfiguration('qos')

    parameters={
          'frame_id':'base_footprint',
          'use_sim_time':use_sim_time,
          'subscribe_rgbd':True,
          'subscribe_scan':True,
          'use_action_for_goal':True,
          'qos_scan':qos,
          'qos_image':qos,
          'qos_imu':qos,
          # RTAB-Map's parameters should be strings:
          'Reg/Strategy':'1',
          'Reg/Force3DoF':'true',
          'RGBD/NeighborLinkRefining':'True',
          'Grid/RangeMin':'0.2', # ignore laser scan points on the robot itself
          'Optimizer/GravitySigma':'0', # Disable imu constraints (we are already in 2D)
          
          # 关键修改：启用视觉里程计并配置发布odom
          'Odom/Strategy':'0',          # 0=视觉里程计 (视觉SLAM), 1=轮式里程计, 2=IMU, 3=融合
          'Odom/ResetCountdown':'1',    # 重置里程计的倒计时（防止异常漂移）
          'Odom/GuessMotion':'true',    # 运动估计优化
          'publish_odom':'true',        # 发布odom话题
          'publish_odom_tf':'true',     # 发布odom相关的TF变换
          'odom_frame_id':'odom',       # 里程计坐标系ID
          'base_frame_id':'base_footprint',  # 机器人基坐标系
          'ground_truth_frame_id':'odom',    # 地面真实坐标系（与odom一致）
          
          # 视觉里程计优化参数
          'Vis/MaxFeatures':'1000',     # 特征点数量
          'Vis/MinFeatures':'100',      # 最小特征点数量
          'RGBD/ProximityBySpace':'true', # 空间邻近性检测
          'RGBD/ProximityPathMaxNeighbors':'10', # 邻近路径最大邻居数
    }

    remappings=[
          ('rgb/image', '/camera/color/image_raw'),
          ('rgb/camera_info', '/camera/color/camera_info'),
          ('depth/image', '/camera/depth/image_raw'),
          # 移除了 ('odom', '/odom') 这一行，不再订阅外部odom
        ]

    return LaunchDescription([

        # Launch arguments
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        
        DeclareLaunchArgument(
            'qos', default_value='2',
            description='QoS used for input sensor topics'),
            
        # 核心RTAB-Map节点（在ROS2中是rtabmap_node，不是rtabmap）
        Node(
            package='rtabmap_ros', executable='rtabmap_node', output='screen',
            parameters=[parameters],
            remappings=remappings,
            arguments=['-d']),  # -d 启用数据库模式
            
        # 可视化节点
        Node(
            package='rtabmap_ros', executable='rtabmap_viz', output='screen',
            parameters=[parameters],
            remappings=remappings),
        
    ])