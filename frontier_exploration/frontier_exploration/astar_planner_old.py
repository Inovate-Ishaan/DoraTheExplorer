import heapq
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener, TransformException


class AStarPlanner(Node):

    def __init__(self):
        super().__init__('astar_planner')

        self.declare_parameter('obstacle_inflation', 0.30)

        self.inflation = float(
            self.get_parameter('obstacle_inflation').value
        )

        self.map_msg = None
        self.goal_msg = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            '/exploration_goal',
            self.goal_callback,
            10
        )

        self.path_pub = self.create_publisher(
            Path,
            '/planned_path',
            10
        )

        self.get_logger().info('A* planner started.')

    def map_callback(self, msg):
        self.map_msg = msg

        if self.goal_msg is not None:
            self.plan_path()

    def goal_callback(self, msg):
        self.goal_msg = msg

        if self.map_msg is not None:
            self.plan_path()

    # Convert world coordinates into occupancy-grid cell coordinates.
    def world_to_grid(self, x, y, info):
        q = info.origin.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        dx = x - info.origin.position.x
        dy = y - info.origin.position.y

        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

        gx = int(math.floor(local_x / info.resolution))
        gy = int(math.floor(local_y / info.resolution))

        return gx, gy

    # Convert a grid cell into its world-coordinate centre.
    def grid_to_world(self, gx, gy, info):
        yaw = math.atan2(
            2.0 * (
                info.origin.orientation.w * info.origin.orientation.z
                + info.origin.orientation.x * info.origin.orientation.y
            ),
            1.0 - 2.0 * (
                info.origin.orientation.y ** 2
                + info.origin.orientation.z ** 2
            )
        )

        local_x = (gx + 0.5) * info.resolution
        local_y = (gy + 0.5) * info.resolution

        x = (
            info.origin.position.x
            + math.cos(yaw) * local_x
            - math.sin(yaw) * local_y
        )

        y = (
            info.origin.position.y
            + math.sin(yaw) * local_x
            + math.cos(yaw) * local_y
        )

        return x, y

    def build_grid(self, msg):
        info = msg.info
        width = info.width
        height = info.height

        grid = np.asarray(msg.data, dtype=np.int16).reshape(
            (height, width)
        )

        # Only confidently free cells are traversable.
        unknown = grid == -1
        free = (grid >= 0) & (grid <= 25)
        occupied = grid >= 65
        uncertain = ~(unknown | free | occupied)

        blocked = unknown | occupied | uncertain

        # Inflate occupied cells to provide clearance for the drone.
        radius_cells = int(math.ceil(
            self.inflation / info.resolution
        ))

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

                inflated[
                    dst_y0:dst_y1,
                    dst_x0:dst_x1
                ] |= occupied[
                    src_y0:src_y1,
                    src_x0:src_x1
                ]

        blocked |= inflated

        return blocked

    @staticmethod
    def heuristic(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])

        # Octile-distance heuristic for 8-directional movement.
        return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)

    def astar(self, blocked, start, goal):
        height, width = blocked.shape

        def valid(cell):
            x, y = cell
            return (
                0 <= x < width
                and 0 <= y < height
                and not blocked[y, x]
            )

        if not valid(start):
            self.get_logger().warning(
                'Drone start cell is blocked or outside the map.'
            )
            return None

        if not valid(goal):
            self.get_logger().warning(
                'Frontier goal cell is blocked or outside the map.'
            )
            return None

        open_heap = []
        heapq.heappush(
            open_heap,
            (self.heuristic(start, goal), 0.0, start)
        )

        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        directions = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (-1, -1, math.sqrt(2.0))
        ]

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)

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

                # Prevent diagonal movement through obstacle corners.
                if dx != 0 and dy != 0:
                    side_a = (current[0] + dx, current[1])
                    side_b = (current[0], current[1] + dy)

                    if not valid(side_a) or not valid(side_b):
                        continue

                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    f_score = (
                        tentative_g + self.heuristic(neighbor, goal)
                    )

                    heapq.heappush(
                        open_heap,
                        (f_score, tentative_g, neighbor)
                    )

        return None

    def plan_path(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                Time()
            )
        except TransformException as ex:
            self.get_logger().warning(
                f'Could not get drone position: {ex}'
            )
            return

        info = self.map_msg.info

        drone_x = transform.transform.translation.x
        drone_y = transform.transform.translation.y

        start = self.world_to_grid(drone_x, drone_y, info)

        goal_position = self.goal_msg.pose.position

        goal = self.world_to_grid(
            goal_position.x,
            goal_position.y,
            info
        )

        blocked = self.build_grid(self.map_msg)

        path_cells = self.astar(blocked, start, goal)

        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        if path_cells is None:
            self.path_pub.publish(path_msg)
            self.get_logger().warning('No valid path found.')
            return

        for gx, gy in path_cells:
            x, y = self.grid_to_world(gx, gy, info)

            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = path_msg.header.stamp

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

        distance = len(path_cells) * info.resolution

        self.get_logger().info(
            f'Path published: {len(path_cells)} cells, '
            f'approx. {distance:.2f} m'
        )


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