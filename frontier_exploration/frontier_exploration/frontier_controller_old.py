#!/usr/bin/env python3

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import TwistStamped
from ardupilot_msgs.msg import Status

import tf2_ros


class FrontierController(Node):

    def __init__(self):
        super().__init__('frontier_controller')

        # Parameters
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('max_speed', 0.35)
        self.declare_parameter('kp', 0.8)
        self.declare_parameter('lookahead', 0.25)
        self.declare_parameter('goal_tolerance', 0.18)
        self.declare_parameter('path_timeout', 8.0)
        self.declare_parameter('dry_run', True)

        self.control_rate = self.get_parameter(
            'control_rate').value

        self.max_speed = self.get_parameter(
            'max_speed').value

        self.kp = self.get_parameter('kp').value

        self.lookahead = self.get_parameter(
            'lookahead').value

        self.goal_tolerance = self.get_parameter(
            'goal_tolerance').value

        self.path_timeout = self.get_parameter(
            'path_timeout').value

        self.dry_run = self.get_parameter('dry_run').value

        # State
        self.path = None
        self.odom = None
        self.status = None

        self.path_received_time = 0.0
        self.odom_received_time = 0.0
        self.status_received_time = 0.0

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self
        )

        # Subscribers
        self.create_subscription(
            Path,
            '/planned_path',
            self.path_callback,
            10
        )

        self.create_subscription(
            Odometry,
            '/odometry',
            self.odom_callback,
            10
        )

        self.create_subscription(
            Status,
            '/ap/v1/status',
            self.status_callback,
            10
        )

        # Publishers
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/ap/v1/cmd_vel',
            10
        )

        # Preview publisher for safe testing
        self.preview_pub = self.create_publisher(
            TwistStamped,
            '/frontier_controller/cmd_vel_preview',
            10
        )

        # Control loop
        self.timer = self.create_timer(
            1.0 / self.control_rate,
            self.control_loop
        )

        self.get_logger().info(
            'Frontier Controller initialized.'
        )

        self.get_logger().info(
            f'Dry run: {self.dry_run}'
        )

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def path_callback(self, msg):
        if not msg.poses:
            self.path = None
            return

        if msg.header.frame_id != 'map':
            self.get_logger().warning(
                f"Expected map frame, received "
                f"{msg.header.frame_id}"
            )
            return

        self.path = msg
        self.path_received_time = time.monotonic()

    def odom_callback(self, msg):
        self.odom = msg
        self.odom_received_time = time.monotonic()

    def status_callback(self, msg):
        self.status = msg
        self.status_received_time = time.monotonic()

    # --------------------------------------------------
    # Transform odometry position into map frame
    # --------------------------------------------------

    def get_map_position(self):

        if self.odom is None:
            return None

        odom_frame = self.odom.header.frame_id

        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                odom_frame,
                Time()
            )

        except Exception:
            return None

        p = self.odom.pose.pose.position
        q = transform.transform.rotation

        # Extract yaw from map -> odom transform
        tf_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        # Transform odometry position into map
        c = math.cos(tf_yaw)
        s = math.sin(tf_yaw)

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y

        map_x = tx + c * p.x - s * p.y
        map_y = ty + s * p.x + c * p.y

        return map_x, map_y

    # --------------------------------------------------
    # Select lookahead waypoint
    # --------------------------------------------------

    def get_target_waypoint(self, x, y):

        points = self.path.poses

        # Find nearest waypoint
        nearest_idx = min(
            range(len(points)),
            key=lambda i: (
                (points[i].pose.position.x - x) ** 2 +
                (points[i].pose.position.y - y) ** 2
            )
        )

        # Move forward along path until lookahead distance
        for i in range(nearest_idx, len(points)):

            px = points[i].pose.position.x
            py = points[i].pose.position.y

            distance = math.hypot(px - x, py - y)

            if distance >= self.lookahead:
                return px, py

        # If no lookahead point exists, use final goal
        final = points[-1].pose.position

        return final.x, final.y

    # --------------------------------------------------
    # Generate velocity command
    # --------------------------------------------------

    def calculate_velocity(self, x, y, target_x, target_y):

        dx = target_x - x
        dy = target_y - y

        vx = self.kp * dx
        vy = self.kp * dy

        speed = math.hypot(vx, vy)

        if speed > self.max_speed:

            scale = self.max_speed / speed

            vx *= scale
            vy *= scale

        return vx, vy

    # --------------------------------------------------
    # Publish command
    # --------------------------------------------------

    def publish_command(self, vx, vy):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()

        # Path and commands are expressed in map frame
        msg.header.frame_id = 'map'

        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        # Always publish preview
        self.preview_pub.publish(msg)

        # Actual commands require explicit activation
        safe_to_control = (
            self.status is not None
            and time.monotonic() - self.status_received_time < 3.0
            and self.status.armed
            and self.status.flying
            and self.status.external_control
            and len(self.status.failsafe) == 0
        )

        if not self.dry_run:
            if safe_to_control:
                self.cmd_pub.publish(msg)
            else:
                stop = TwistStamped()
                stop.header.stamp = msg.header.stamp
                stop.header.frame_id = 'map'
                self.cmd_pub.publish(stop)

    # --------------------------------------------------
    # Main control loop
    # --------------------------------------------------

    def control_loop(self):

        now = time.monotonic()

        # Validate odometry
        if (
            self.odom is None
            or now - self.odom_received_time > 0.5
        ):
            self.publish_command(0.0, 0.0)
            return

        # Validate path
        if (
            self.path is None
            or now - self.path_received_time > self.path_timeout
        ):
            self.publish_command(0.0, 0.0)
            return

        # Get current position in map
        position = self.get_map_position()

        if position is None:
            self.publish_command(0.0, 0.0)
            return

        x, y = position

        # Get final goal
        final = self.path.poses[-1].pose.position

        goal_distance = math.hypot(
            final.x - x,
            final.y - y
        )

        # Goal reached
        if goal_distance < self.goal_tolerance:

            self.publish_command(0.0, 0.0)

            self.get_logger().info(
                'Frontier reached!',
                throttle_duration_sec=5.0
            )

            return

        # Select lookahead waypoint
        target_x, target_y = self.get_target_waypoint(x, y)

        # Calculate velocity
        vx, vy = self.calculate_velocity(
            x, y, target_x, target_y
        )

        self.publish_command(vx, vy)


def main(args=None):

    rclpy.init(args=args)

    node = FrontierController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.publish_command(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()