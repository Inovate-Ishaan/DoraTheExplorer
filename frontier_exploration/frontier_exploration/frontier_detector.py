import rclpy
import numpy as np

from collections import deque

from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray


class FrontierDetector(Node):

    def __init__(self):
        super().__init__('frontier_detector')

        # Subscribe to the live SLAM map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        # Publish detected frontier clusters
        self.frontier_pub = self.create_publisher(
            MarkerArray,
            '/frontiers',
            10
        )

        self.get_logger().info('Frontier Detector Started')

    def map_callback(self, msg):

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        # Convert occupancy grid into a 2D NumPy array
        grid = np.array(msg.data, dtype=np.int8).reshape(
            (height, width)
        )

        # Free cells: occupancy values 0-24
        free = (grid >= 0) & (grid <= 24)

        # Unknown cells have occupancy value -1
        unknown = (grid == -1)

        # Identify free cells adjacent to unknown cells
        adjacent_unknown = np.zeros_like(unknown, dtype=bool)

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:

                if dx == 0 and dy == 0:
                    continue

                shifted = np.zeros_like(unknown)

                src_y = slice(max(0, -dy), min(height, height - dy))
                src_x = slice(max(0, -dx), min(width, width - dx))

                dst_y = slice(max(0, dy), min(height, height + dy))
                dst_x = slice(max(0, dx), min(width, width + dx))

                shifted[dst_y, dst_x] = unknown[src_y, src_x]

                adjacent_unknown |= shifted

        # Frontier = free cell adjacent to unknown cell
        frontier_mask = free & adjacent_unknown

        frontier_cells = set(
            map(tuple, np.argwhere(frontier_mask))
        )

        # Find connected frontier clusters using BFS
        clusters = []

        while frontier_cells:

            start = frontier_cells.pop()

            queue = deque([start])
            cluster = [start]

            while queue:

                y, x = queue.popleft()

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:

                        if dx == 0 and dy == 0:
                            continue

                        neighbor = (y + dy, x + dx)

                        if neighbor in frontier_cells:

                            frontier_cells.remove(neighbor)

                            queue.append(neighbor)
                            cluster.append(neighbor)

            # Ignore very small frontier clusters
            if len(cluster) >= 5:
                clusters.append(cluster)

        # Publish frontier cluster centroids
        marker_array = MarkerArray()

        for i, cluster in enumerate(clusters):

            rows = [cell[0] for cell in cluster]
            cols = [cell[1] for cell in cluster]

            center_x = sum(cols) / len(cols)
            center_y = sum(rows) / len(rows)

            # Grid coordinates -> world coordinates
            origin = msg.info.origin

            local_x = (center_x + 0.5) * resolution
            local_y = (center_y + 0.5) * resolution

            q = origin.orientation

            sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
            cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

            yaw = np.arctan2(sin_yaw, cos_yaw)

            world_x = (
                origin.position.x
                + local_x * np.cos(yaw)
                - local_y * np.sin(yaw)
            )

            world_y = (
                origin.position.y
                + local_x * np.sin(yaw)
                + local_y * np.cos(yaw)
            )

            marker = Marker()

            marker.header = msg.header
            marker.ns = 'frontier_clusters'
            marker.id = i

            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.5

            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.20
            marker.scale.y = 0.20
            marker.scale.z = 0.20

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        # Delete markers from previous map updates
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.insert(0, delete_marker)

        self.frontier_pub.publish(marker_array)

        self.get_logger().info(
            f'Detected {len(clusters)} frontier clusters'
        )


def main(args=None):

    rclpy.init(args=args)

    node = FrontierDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()