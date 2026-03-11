#!/usr/bin/env python3
# x3_nav_controller.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
# ROS2 原生消息导入（关键：导入Quaternion类）
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
# ROS2 原生QoS配置
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
# 数学库（仅内置）
from math import atan2, pi, sin, cos
import threading
import time

class YahboomX3Controller(Node):
    """亚博X3小车导航控制类（终极稳定版：修复姿态获取为0的问题）"""
    def __init__(self, node_name="x3_nav_controller"):
        super().__init__(node_name)
        
        # 姿态数据缓存（线程安全）
        self.lock = threading.Lock()
        self.current_odom_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}  # 里程计姿态
        self.current_amcl_pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}  # AMCL定位姿态
        
        # 1. 订阅里程计话题（高频更新）
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10  # QoS深度，匹配小车发布频率
        )
        
        # 2. 订阅AMCL定位话题（高精度，低频）
        amcl_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_callback,
            amcl_qos
        )
        
        # 3. 创建Nav2动作客户端（推荐方式）
        self.nav_action_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        
        # 等待动作服务器上线（最多等待5秒）
        self.get_logger().info("等待Nav2导航服务器上线...")
        if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Nav2服务器未响应，仅能使用话题方式发送目标点")
            self.nav_action_client = None

        # 新增：spin线程（自动启动，确保回调执行）
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

    def _spin_loop(self):
        """内部spin循环，驱动回调执行"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)  # 非阻塞spin，每0.1秒检查一次

    def quaternion_to_yaw(self, quaternion):
        """ROS2原生实现：四元数转偏航角（yaw）"""
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w
        
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = atan2(siny_cosp, cosy_cosp)
        
        # 确保yaw在[-π, π]范围内
        if yaw > pi:
            yaw -= 2 * pi
        elif yaw < -pi:
            yaw += 2 * pi
        return yaw

    def yaw_to_quaternion(self, yaw):
        """
        关键修复：返回ROS2原生的Quaternion对象
        :param yaw: 偏航角（弧度）
        :return: geometry_msgs.msg.Quaternion（官方类型）
        """
        quat = Quaternion()  # 使用ROS2原生Quaternion类
        half_yaw = yaw / 2.0
        quat.x = 0.0
        quat.y = 0.0
        quat.z = sin(half_yaw)
        quat.w = cos(half_yaw)
        return quat

    def odom_callback(self, msg: Odometry):
        """里程计话题回调，更新实时姿态"""
        with self.lock:
            # 新增日志：验证回调是否执行
            self.get_logger().debug(f"收到里程计数据：x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}")
            self.current_odom_pose["x"] = msg.pose.pose.position.x
            self.current_odom_pose["y"] = msg.pose.pose.position.y
            self.current_odom_pose["yaw"] = self.quaternion_to_yaw(msg.pose.pose.orientation)

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        """AMCL定位回调，更新高精度姿态"""
        with self.lock:
            # 新增日志：验证回调是否执行
            self.get_logger().debug(f"收到AMCL数据：x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}")
            self.current_amcl_pose["x"] = msg.pose.pose.position.x
            self.current_amcl_pose["y"] = msg.pose.pose.position.y
            self.current_amcl_pose["yaw"] = self.quaternion_to_yaw(msg.pose.pose.orientation)

    def get_current_pose(self, use_amcl=True):
        """获取当前小车姿态"""
        with self.lock:
            if use_amcl:
                return self.current_amcl_pose.copy()
            else:
                return self.current_odom_pose.copy()

    def send_goal_by_topic(self, x, y, yaw=0.0, frame_id="map"):
        """通过/goal_pose话题发送导航目标（备用方式）"""
        goal_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        goal_pub = self.create_publisher(PoseStamped, "/goal_pose", goal_qos)
        
        # 构造目标姿态（全ROS2原生类型）
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = frame_id
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation = self.yaw_to_quaternion(yaw)
        
        # 发布目标（重复发布3次确保接收）
        for _ in range(3):
            goal_pub.publish(goal_msg)
            time.sleep(0.1)
        
        self.get_logger().info(f"已通过话题发送目标点：x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        self.destroy_publisher(goal_pub)

    def send_goal_by_action(self, x, y, yaw=0.0, frame_id="map"):
        """通过NavigateToPose动作发送导航目标（推荐方式）"""
        if not self.nav_action_client:
            self.get_logger().error("导航动作服务器不可用，切换到话题方式发送")
            self.send_goal_by_topic(x, y, yaw, frame_id)
            return False
        
        # 构造动作目标（全ROS2原生类型）
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = frame_id
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation = self.yaw_to_quaternion(yaw)
        
        # 构造动作请求
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        # 发送目标
        self.get_logger().info(f"发送导航目标：x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        send_goal_future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav_feedback_callback
        )
        
        # 等待发送结果
        send_goal_future.add_done_callback(self.nav_goal_response_callback)
        return True

    def nav_feedback_callback(self, feedback_msg):
        """导航过程反馈（实时显示剩余距离）"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f"导航中，剩余距离：{feedback.distance_remaining:.2f} 米")

    def nav_goal_response_callback(self, future):
        """目标发送结果回调"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("导航目标被拒绝！")
            return
        
        self.get_logger().info("导航目标已接受，开始导航...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        """导航完成结果回调"""
        result = future.result().result
        self.get_logger().info(f"导航完成！结果：{result}")

    def cancel_navigation(self):
        """取消当前导航任务"""
        if self.nav_action_client and self.nav_action_client.server_is_ready():
            cancel_goal = NavigateToPose.Goal()
            cancel_goal.cancel_goal = True
            self.nav_action_client.send_goal_async(cancel_goal)
            self.get_logger().info("已发送取消导航请求")
        else:
            self.get_logger().warn("无法取消导航：动作服务器不可用")

# ------------------- 测试示例（最终稳定版） -------------------
def main(args=None):
    # 检查并初始化ROS2 Context（避免重复初始化）
    if not rclpy.ok():
        rclpy.init(args=args)
    
    # 创建控制类实例
    x3_controller = YahboomX3Controller()
    
    # 1. 实时打印小车姿态（每1秒打印一次）
    def print_pose_loop():
        while rclpy.ok():
            odom_pose = x3_controller.get_current_pose(use_amcl=False)
            amcl_pose = x3_controller.get_current_pose(use_amcl=True)
            x3_controller.get_logger().info(
                f"\n里程计姿态：x={odom_pose['x']:.2f}, y={odom_pose['y']:.2f}, yaw={odom_pose['yaw']:.2f}"
                f"\nAMCL姿态：x={amcl_pose['x']:.2f}, y={amcl_pose['y']:.2f}, yaw={amcl_pose['yaw']:.2f}"
            )
            time.sleep(1.0)
    
    # 启动打印线程
    pose_thread = threading.Thread(target=print_pose_loop, daemon=True)
    pose_thread.start()
    
    # 2. 发送安全目标点（x=0.0, y=0.0，朝向90度）
    x3_controller.send_goal_by_action(x=0.0, y=0.0, yaw=1.57)
    
    # 3. 运行节点（阻塞直到手动退出）
    try:
        while rclpy.ok():
            time.sleep(0.1)  # 保持主线程存活
    except KeyboardInterrupt:
        x3_controller.get_logger().info("用户中断，退出程序")
        x3_controller.cancel_navigation()
    finally:
        # 优雅销毁节点和关闭Context
        x3_controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()