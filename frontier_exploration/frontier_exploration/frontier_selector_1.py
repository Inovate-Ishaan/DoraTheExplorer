import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener, TransformException


class FrontierSelector(Node):
    """
    Picks the nearest frontier, but:
      * commits to the current goal while it still exists (no flip-flop),
      * blacklists a goal once reached or if it takes too long,
        so unreachable / un-clearable frontiers can't stall exploration.
    """

    def __init__(self):
        super().__init__('frontier_selector')

        self.declare_parameter('reach_radius', 0.5)
        self.declare_parameter('blacklist_radius', 0.6)
        self.declare_parameter('blacklist_time', 120.0)
        self.declare_parameter('goal_timeout', 45.0)
        self.declare_parameter('same_goal_radius', 0.6)

        self.reach_radius = self.get_parameter('reach_radius').value
        self.blacklist_radius = self.get_parameter(
            'blacklist_radius').value
        self.blacklist_time = self.get_parameter('blacklist_time').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.same_goal_radius = self.get_parameter(
            'same_goal_radius').value

        self.current = None          # (x, y)
        self.current_since = 0.0
        self.blacklist = []          # (x, y, expiry)

        self.frontier_sub = self.create_subscription(
            MarkerArray, '/frontiers', self.frontier_callback, 10)
        self.goal_pub = self.create_publisher(
            PoseStamped, '/exploration_goal', 10)
        self.selected_pub = self.create_publisher(
            Marker, '/selected_frontier', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info('Frontier Selector Started')

    def is_blacklisted(self, x, y, now):
        self.blacklist = [b for b in self.blacklist if b[2] > now]
        return any(math.hypot(x - bx, y - by) < self.blacklist_radius
                   for bx, by, _ in self.blacklist)

    def add_blacklist(self, x, y, now, reason):
        self.blacklist.append((x, y, now + self.blacklist_time))
        self.get_logger().info(
            f'Blacklisted ({x:.2f}, {y:.2f}): {reason}')

    def frontier_callback(self, msg):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time())
        except TransformException as ex:
            self.get_logger().warning(f'Could not get drone pose: {ex}')
            return

        now = time.monotonic()
        drone_x = transform.transform.translation.x
        drone_y = transform.transform.translation.y

        frontiers = [
            m for m in msg.markers
            if m.action == Marker.ADD and m.type == Marker.SPHERE
        ]

        if not frontiers:
            self.get_logger().info('No frontiers available')
            self.current = None
            return

        # Retire current goal if reached or timed out
        if self.current is not None:
            cx, cy = self.current
            if math.hypot(cx - drone_x, cy - drone_y) < self.reach_radius:
                self.add_blacklist(cx, cy, now, 'reached')
                self.current = None
            elif now - self.current_since > self.goal_timeout:
                self.add_blacklist(cx, cy, now, 'timeout')
                self.current = None

        candidates = [
            f for f in frontiers
            if not self.is_blacklisted(
                f.pose.position.x, f.pose.position.y, now)
        ]

        if not candidates:
            self.get_logger().info(
                'All remaining frontiers blacklisted')
            return

        best = None

        # Keep committed goal if a matching frontier still exists
        if self.current is not None:
            cx, cy = self.current
            matches = [
                f for f in candidates
                if math.hypot(f.pose.position.x - cx,
                              f.pose.position.y - cy)
                < self.same_goal_radius
            ]
            if matches:
                best = min(matches, key=lambda f: math.hypot(
                    f.pose.position.x - cx, f.pose.position.y - cy))

        # Otherwise pick nearest candidate
        if best is None:
            best = min(candidates, key=lambda f: math.hypot(
                f.pose.position.x - drone_x,
                f.pose.position.y - drone_y))
            self.current_since = now

        self.current = (best.pose.position.x, best.pose.position.y)
        best_distance = math.hypot(
            self.current[0] - drone_x, self.current[1] - drone_y)

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = 'map'
        goal.pose.position.x = self.current[0]
        goal.pose.position.y = self.current[1]
        goal.pose.position.z = 0.0

        yaw = math.atan2(self.current[1] - drone_y,
                         self.current[0] - drone_x)
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pub.publish(goal)

        selected = Marker()
        selected.header = goal.header
        selected.ns = 'selected_frontier'
        selected.id = 0
        selected.type = Marker.SPHERE
        selected.action = Marker.ADD
        selected.pose = best.pose
        selected.scale.x = selected.scale.y = selected.scale.z = 0.35
        selected.color.r = 1.0
        selected.color.a = 1.0
        self.selected_pub.publish(selected)

        self.get_logger().info(
            f'Selected frontier at ({self.current[0]:.2f}, '
            f'{self.current[1]:.2f}), distance = {best_distance:.2f} m',
            throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
