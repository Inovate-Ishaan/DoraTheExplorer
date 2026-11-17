
import heapq
import math
from collections import deque
 
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
 
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException
 
try:
    from scipy import ndimage
except ImportError:  # fall back to a pure-python flood fill
    ndimage = None
 
 
class AStarPlanner(Node):
 
    def __init__(self):
        super().__init__('astar_planner')
 
        self.declare_parameter('obstacle_inflation', 0.60)
        self.declare_parameter('start_snap_radius', 0.6)
        self.declare_parameter('goal_snap_radius', 1.0)
 
        self.inflation = float(
            self.get_parameter('obstacle_inflation').value)
        self.start_snap = float(
            self.get_parameter('start_snap_radius').value)
        self.goal_snap = float(
            self.get_parameter('goal_snap_radius').value)
 
        self.map_msg = None
        self.goal_msg = None
 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
 
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/exploration_goal', self.goal_callback, 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        # Tells the frontier selector that a goal cannot be reached, so it
        # can blacklist it immediately instead of waiting for its timeout.
        self.unreachable_pub = self.create_publisher(
            PoseStamped, '/unreachable_goal', 10)
 
        self.get_logger().info('A* planner started.')
 
    def map_callback(self, msg):
        self.map_msg = msg
        if self.goal_msg is not None:
            self.plan_path()
 
    def goal_callback(self, msg):
        self.goal_msg = msg
        if self.map_msg is not None:
            self.plan_path()
 
    @staticmethod
    def _yaw(q):
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
 
    def world_to_grid(self, x, y, info):
        yaw = self._yaw(info.origin.orientation)
 
        dx = x - info.origin.position.x
        dy = y - info.origin.position.y
 
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
 
        return (int(math.floor(local_x / info.resolution)),
                int(math.floor(local_y / info.resolution)))
 
    def grid_to_world(self, gx, gy, info):
        yaw = self._yaw(info.origin.orientation)
 
        local_x = (gx + 0.5) * info.resolution
        local_y = (gy + 0.5) * info.resolution
 
        x = (info.origin.position.x
             + math.cos(yaw) * local_x - math.sin(yaw) * local_y)
        y = (info.origin.position.y
             + math.sin(yaw) * local_x + math.cos(yaw) * local_y)
        return x, y
 
    def build_grid(self, msg):
        info = msg.info
        width = info.width
        height = info.height
 
        grid = np.asarray(msg.data, dtype=np.int16).reshape(
            (height, width))
 
        unknown = grid == -1
        free = (grid >= 0) & (grid <= 25)
        occupied = grid >= 65
        uncertain = ~(unknown | free | occupied)
 
        blocked = unknown | occupied | uncertain
 
        radius_cells = int(math.ceil(self.inflation / info.resolution))
        inflated = np.zeros_like(occupied, dtype=bool)
 
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
 
                src_x0 = max(0, -dx)
                src_x1 = min(width, width - dx)
                src_y0 = max(0, -dy)
                src_y1 = min(height, height - dy)
 
                dst_x0 = max(0, dx)
                dst_x1 = min(width, width + dx)
                dst_y0 = max(0, dy)
                dst_y1 = min(height, height + dy)
 
                inflated[dst_y0:dst_y1, dst_x0:dst_x1] |= \
                    occupied[src_y0:src_y1, src_x0:src_x1]
 
        blocked |= inflated
        return blocked
 
    @staticmethod
    def nearest_allowed(allowed, cell, max_radius):
        """Nearest cell where `allowed` is True within max_radius cells,
        else None. Returns `cell` itself if it is already allowed."""
        height, width = allowed.shape
        cx, cy = cell
 
        if 0 <= cx < width and 0 <= cy < height and allowed[cy, cx]:
            return cell
 
        x0 = max(0, cx - max_radius)
        x1 = min(width, cx + max_radius + 1)
        y0 = max(0, cy - max_radius)
        y1 = min(height, cy + max_radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None
 
        window = allowed[y0:y1, x0:x1]
        idx = np.argwhere(window)
        if idx.size == 0:
            return None
 
        ys = idx[:, 0] + y0
        xs = idx[:, 1] + x0
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        k = int(np.argmin(d2))
        if d2[k] > max_radius * max_radius:
            return None
        return (int(xs[k]), int(ys[k]))
 
    @staticmethod
    def reachable_from(blocked, start):
        """Boolean mask of the free cells connected to `start`.
 
        Uses 4-connectivity, which is exactly the reachability A* has
        (its diagonal moves are only allowed when both neighbouring
        orthogonal cells are free). So if a goal is inside this mask,
        A* is guaranteed to find a path.
        """
        if ndimage is not None:
            labels, _ = ndimage.label(~blocked)
            return labels == labels[start[1], start[0]]
 
        height, width = blocked.shape
        seen = np.zeros_like(blocked, dtype=bool)
        seen[start[1], start[0]] = True
        queue = deque([start])
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y),
                           (x, y + 1), (x, y - 1)):
                if (0 <= nx < width and 0 <= ny < height
                        and not blocked[ny, nx] and not seen[ny, nx]):
                    seen[ny, nx] = True
                    queue.append((nx, ny))
        return seen
 
    @staticmethod
    def heuristic(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)
 
    def astar(self, blocked, start, goal):
        height, width = blocked.shape
 
        def valid(cell):
            x, y = cell
            return (0 <= x < width and 0 <= y < height
                    and not blocked[y, x])
 
        if not valid(start) or not valid(goal):
            return None
 
        open_heap = []
        heapq.heappush(open_heap,
                       (self.heuristic(start, goal), 0.0, start))
 
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
 
        directions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]
 
        while open_heap:
            _, _, current = heapq.heappop(open_heap)
 
            if current in closed:
                continue
 
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
 
            closed.add(current)
 
            for dx, dy, move_cost in directions:
                neighbor = (current[0] + dx, current[1] + dy)
 
                if not valid(neighbor) or neighbor in closed:
                    continue
 
                if dx != 0 and dy != 0:
                    if (not valid((current[0] + dx, current[1]))
                            or not valid((current[0], current[1] + dy))):
                        continue
 
                tentative_g = g_score[current] + move_cost
 
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(
                        open_heap,
                        (tentative_g + self.heuristic(neighbor, goal),
                         tentative_g, neighbor))
 
        return None
 
    def report_unreachable(self, path_msg, reason):
        """Publish an empty path and tell the selector to drop the goal."""
        self.path_pub.publish(path_msg)
        self.unreachable_pub.publish(self.goal_msg)
        self.get_logger().warning(reason, throttle_duration_sec=3.0)
 
    def plan_path(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time())
        except TransformException as ex:
            self.get_logger().warning(
                f'Could not get drone position: {ex}')
            return
 
        info = self.map_msg.info
 
        drone_x = transform.transform.translation.x
        drone_y = transform.transform.translation.y
 
        start = self.world_to_grid(drone_x, drone_y, info)
        gp = self.goal_msg.pose.position
        goal = self.world_to_grid(gp.x, gp.y, info)
 
        blocked = self.build_grid(self.map_msg)
 
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
 
        # Snap the start to the nearest traversable cell: the drone may
        # sit inside the inflation zone of a wall.
        start_r = int(math.ceil(self.start_snap / info.resolution))
        start_s = self.nearest_allowed(~blocked, start, start_r)
        if start_s is None:
            self.path_pub.publish(path_msg)
            self.get_logger().warning(
                f'No free cell within {self.start_snap:.1f} m of the '
                f'drone at ({drone_x:.2f}, {drone_y:.2f}).',
                throttle_duration_sec=3.0)
            return
 
        # Snap the goal only to cells that are CONNECTED to the drone.
        # Snapping to the nearest free cell regardless of connectivity
        # lands in pockets behind walls / closed-off gaps and A* can
        # never reach them.
        reachable = self.reachable_from(blocked, start_s)
        goal_r = int(math.ceil(self.goal_snap / info.resolution))
        goal_s = self.nearest_allowed(reachable, goal, goal_r)
        if goal_s is None:
            self.report_unreachable(
                path_msg,
                f'Goal ({gp.x:.2f}, {gp.y:.2f}) has no cell connected to '
                f'the drone within {self.goal_snap:.1f} m '
                f'(drone region = {int(reachable.sum())} free cells). '
                f'Reported as unreachable.')
            return
 
        path_cells = self.astar(blocked, start_s, goal_s)
 
        if path_cells is None:
            self.report_unreachable(
                path_msg,
                f'A* failed although goal cell {goal_s} is in the '
                f'drone region (start {start_s}). Reported as '
                f'unreachable.')
            return
 
        for gx, gy in path_cells:
            x, y = self.grid_to_world(gx, gy, info)
 
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
 
        self.path_pub.publish(path_msg)
 
        goal_shift = math.hypot(goal_s[0] - goal[0],
                                goal_s[1] - goal[1]) * info.resolution
        self.get_logger().info(
            f'Path published: {len(path_cells)} cells, '
            f'approx. {len(path_cells) * info.resolution:.2f} m '
            f'(goal moved {goal_shift:.2f} m to a reachable cell)',
            throttle_duration_sec=2.0)
 
 
def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()
 