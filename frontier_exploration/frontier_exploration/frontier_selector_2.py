
import math
import time
 
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
 
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
 
from tf2_ros import Buffer, TransformListener, TransformException
 
 
class FrontierSelector(Node):
    """
    Picks the nearest frontier, but:
      * prefers frontiers that keep clear of walls (clearance measured
        on the /map occupancy grid),
      * commits to the current goal while it still exists (no flip-flop),
      * blacklists a goal once reached, if it takes too long, or if the
        planner reports it unreachable (/unreachable_goal).
    """
 
    def __init__(self):
        super().__init__('frontier_selector')
 
        self.declare_parameter('reach_radius', 0.5)
        self.declare_parameter('blacklist_radius', 0.6)
        self.declare_parameter('blacklist_time', 120.0)
        self.declare_parameter('goal_timeout', 45.0)
        self.declare_parameter('same_goal_radius', 0.6)
 
        # Wall avoidance
        # Frontiers closer than this to an occupied cell are skipped
        # while any wall-free frontier exists.
        self.declare_parameter('min_wall_clearance', 0.7)
        # Clearance is only measured up to this distance (metres).
        self.declare_parameter('clearance_cap', 1.0)
        # Score = distance - weight * clearance. 0 = pure nearest-first.
        self.declare_parameter('clearance_weight', 1.0)
        # How long a goal the planner called unreachable stays excluded.
        self.declare_parameter('unreachable_blacklist_time', 30.0)
 
        self.reach_radius = self.get_parameter('reach_radius').value
        self.blacklist_radius = self.get_parameter(
            'blacklist_radius').value
        self.blacklist_time = self.get_parameter('blacklist_time').value
        self.goal_timeout = self.get_parameter('goal_timeout').value
        self.same_goal_radius = self.get_parameter(
            'same_goal_radius').value
        self.min_clearance = float(
            self.get_parameter('min_wall_clearance').value)
        self.clearance_cap = float(
            self.get_parameter('clearance_cap').value)
        self.clearance_weight = float(
            self.get_parameter('clearance_weight').value)
        self.unreachable_time = float(
            self.get_parameter('unreachable_blacklist_time').value)
 
        self.current = None          # (x, y)
        self.current_since = 0.0
        self.blacklist = []          # (x, y, expiry)
 
        self.map_msg = None
        self.grid = None             # int16 array [height, width]
 
        self.frontier_sub = self.create_subscription(
            MarkerArray, '/frontiers', self.frontier_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.unreachable_sub = self.create_subscription(
            PoseStamped, '/unreachable_goal',
            self.unreachable_callback, 10)
        self.goal_pub = self.create_publisher(
            PoseStamped, '/exploration_goal', 10)
        self.selected_pub = self.create_publisher(
            Marker, '/selected_frontier', 10)
 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
 
        self.get_logger().info('Frontier Selector Started')
 
    # --------------------------------------------------
    # Map / clearance
    # --------------------------------------------------
 
    def map_callback(self, msg):
        self.map_msg = msg
        self.grid = np.asarray(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))
 
    def clearance(self, x, y):
        """Distance (m) from (x, y) to the nearest occupied cell,
        capped at clearance_cap. Assumes an axis-aligned map origin
        (slam_toolbox publishes yaw = 0). With no map yet, returns the
        cap, i.e. no penalty."""
        if self.grid is None:
            return self.clearance_cap
 
        info = self.map_msg.info
        res = info.resolution
        cx = int(math.floor((x - info.origin.position.x) / res))
        cy = int(math.floor((y - info.origin.position.y) / res))
        r = int(math.ceil(self.clearance_cap / res))
 
        x0 = max(0, cx - r)
        x1 = min(info.width, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(info.height, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            return self.clearance_cap
 
        idx = np.argwhere(self.grid[y0:y1, x0:x1] >= 65)
        if idx.size == 0:
            return self.clearance_cap
 
        dx = idx[:, 1] + x0 - cx
        dy = idx[:, 0] + y0 - cy
        dist = float(np.sqrt(np.min(dx * dx + dy * dy))) * res
        return min(dist, self.clearance_cap)
 
    # --------------------------------------------------
    # Blacklist
    # --------------------------------------------------
 
    def is_blacklisted(self, x, y, now):
        self.blacklist = [b for b in self.blacklist if b[2] > now]
        return any(math.hypot(x - bx, y - by) < self.blacklist_radius
                   for bx, by, _ in self.blacklist)
 
    def add_blacklist(self, x, y, now, reason, duration=None):
        if duration is None:
            duration = self.blacklist_time
        self.blacklist.append((x, y, now + duration))
        self.get_logger().info(
            f'Blacklisted ({x:.2f}, {y:.2f}): {reason}')
 
    def unreachable_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        now = time.monotonic()
 
        if not self.is_blacklisted(x, y, now):
            self.add_blacklist(x, y, now, 'planner says unreachable',
                               self.unreachable_time)
 
        # Drop the committed goal if it is the one that failed
        if (self.current is not None
                and math.hypot(self.current[0] - x,
                               self.current[1] - y)
                < self.same_goal_radius):
            self.current = None
 
    # --------------------------------------------------
    # Selection
    # --------------------------------------------------
 
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
 
        # Wall clearance of every candidate
        clear = {
            id(f): self.clearance(f.pose.position.x, f.pose.position.y)
            for f in candidates
        }
 
        # Prefer frontiers away from walls. If every frontier is close
        # to a wall, fall back to all of them so exploration never
        # stalls (the planner will blacklist the unreachable ones).
        open_ones = [f for f in candidates
                     if clear[id(f)] >= self.min_clearance]
        if open_ones:
            pool = open_ones
        else:
            pool = candidates
            self.get_logger().info(
                'All frontiers are close to walls; using them anyway.',
                throttle_duration_sec=5.0)
 
        best = None
 
        # Keep committed goal if a matching frontier still exists
        if self.current is not None:
            cx, cy = self.current
            matches = [
                f for f in pool
                if math.hypot(f.pose.position.x - cx,
                              f.pose.position.y - cy)
                < self.same_goal_radius
            ]
            if matches:
                best = min(matches, key=lambda f: math.hypot(
                    f.pose.position.x - cx, f.pose.position.y - cy))
 
        # Otherwise pick the best-scoring candidate: near, and open
        if best is None:
            best = min(pool, key=lambda f: (
                math.hypot(f.pose.position.x - drone_x,
                           f.pose.position.y - drone_y)
                - self.clearance_weight * clear[id(f)]))
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
            f'{self.current[1]:.2f}), distance = {best_distance:.2f} m, '
            f'wall clearance = {clear[id(best)]:.2f} m',
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
 