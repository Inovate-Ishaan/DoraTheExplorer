#!/usr/bin/env python3
"""
Mission manager: explore, then come back and land on the ArUco marker the
drone took off from.

    WAIT_FLYING -> HOME_ACQUIRE -> EXPLORE -> RETURN -> (SEARCH) -> HANDOFF
                                                                   -> LANDING -> DONE

 WAIT_FLYING   wait until the frontier_controller has armed and taken off.
 HOME_ACQUIRE  hover and measure where the marker is in the map frame (median
               of several detections). Falls back to the takeoff point.
 EXPLORE       frontier_selector + planner + controller do the work. Ends when
               no usable frontier is left (for a few seconds), on the time
               limit, or on /mission/return_home.
 RETURN        send the planner the home position; the controller follows.
               As soon as the marker is seen near home -> HANDOFF.
 SEARCH        marker not visible on arrival: visit a square spiral around
               home until it is.
 HANDOFF       precision_lander takes over (centre, descend, LAND mode).
               If it loses the marker for too long, go back to SEARCH.
 LANDING/DONE  LAND mode is running / the vehicle has disarmed.

Who may steer in each state is set through two latched gates:
    /mission/selector_enable   frontier_selector (EXPLORE only)
    /mission/controller_enable frontier_controller (EXPLORE, RETURN, SEARCH)
In HANDOFF/LANDING neither is on, so precision_lander is the only publisher
on /ap/v1/cmd_vel.

Manual commands (std_srvs/Trigger):
    /mission/return_home   stop exploring now, fly back to the marker and land
    /mission/land_here     LAND immediately, where the drone is
"""

import math
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger
from ardupilot_msgs.msg import Status
from ardupilot_msgs.srv import ModeSwitch

import tf2_ros


def _now():
    return time.monotonic()


def spiral_points(step, radius):
    """Square spiral of (dx, dy) offsets around (0, 0), spacing `step`,
    up to `radius` (Chebyshev)."""
    max_n = max(1, int(radius / step))
    pts = [(0.0, 0.0)]
    x = y = 0
    dx, dy = 1, 0
    seg = 1
    while True:
        for _ in range(2):
            for _ in range(seg):
                x += dx
                y += dy
                if max(abs(x), abs(y)) > max_n:
                    return pts
                pts.append((x * step, y * step))
            dx, dy = -dy, dx
        seg += 1


GATES = {
    # state: (selector_enable, controller_enable)
    'WAIT_FLYING': (False, False),
    'HOME_ACQUIRE': (False, False),
    'EXPLORE': (True, True),
    'RETURN': (False, True),
    'SEARCH': (False, True),
    'HANDOFF': (False, False),
    'LANDING': (False, False),
    'DONE': (False, False),
}


class MissionManager(Node):

    def __init__(self):
        super().__init__('mission_manager')

        dp = self.declare_parameter
        # Home / marker
        dp('home_acquire_alt', 1.5)     # m: start measuring above this height
        dp('home_samples', 10)
        dp('home_timeout', 10.0)        # s of measuring before giving up
        # Exploration end
        dp('min_explore_time', 20.0)    # s: ignore "no frontiers" before this
        dp('no_frontier_confirm', 8.0)  # s with nothing left => finished
        dp('frontier_stale', 20.0)      # s without selector news => nothing left
        dp('max_explore_time', 1800.0)  # s hard limit, 0 = off
        # Return / search
        dp('handoff_radius', 3.0)       # marker seen within this of home
        dp('arrive_radius', 0.8)
        dp('arrive_dwell', 3.0)
        dp('search_step', 1.0)
        dp('search_radius', 3.0)
        dp('search_point_radius', 0.4)
        dp('search_point_time', 12.0)
        dp('lander_lost_timeout', 20.0)
        dp('land_mode', 9)              # Copter LAND
        dp('mode_service', '/ap/v1/mode_switch')
        dp('status_topic', '/ap/v1/status')
        dp('tick_rate', 5.0)

        g = lambda n: self.get_parameter(n).value
        self.p = {n: g(n) for n in (
            'home_acquire_alt', 'home_samples', 'home_timeout',
            'min_explore_time', 'no_frontier_confirm', 'frontier_stale',
            'max_explore_time', 'handoff_radius', 'arrive_radius',
            'arrive_dwell', 'search_step', 'search_radius',
            'search_point_radius', 'search_point_time',
            'lander_lost_timeout', 'land_mode')}

        # ---------------- state ----------------
        self.state = 'WAIT_FLYING'
        self.state_since = _now()
        self.status = None
        self.status_time = 0.0
        self.was_flying = False
        self.odom_z = None

        self.takeoff_pos = None
        self.home = None
        self.home_source = ''
        self.samples = []
        self.acquire_start = None

        self.last_marker_time = -math.inf
        self.last_marker_map = None

        self.remaining = None
        self.remaining_time = None
        self.none_since = None
        self.explore_start = None

        self.arrive_since = None
        self.search_pts = []
        self.search_idx = 0
        self.search_idx_since = 0.0
        self.unreachable = []          # (x, y, t)
        self.handoff_called = False
        self.need_land_mode = False
        self.last_goal_pub = 0.0
        self.last_gate_pub = 0.0
        self.last_warn = 0.0

        # ---------------- ROS ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE)
        self.selector_gate = self.create_publisher(
            Bool, '/mission/selector_enable', latched)
        self.controller_gate = self.create_publisher(
            Bool, '/mission/controller_enable', latched)
        self.goal_pub = self.create_publisher(
            PoseStamped, '/exploration_goal', 10)

        self.create_subscription(
            PoseStamped, '/precision_landing/marker_pose',
            self.marker_cb, 10)
        self.create_subscription(
            Int32, '/frontier_selector/remaining', self.remaining_cb, 10)
        self.create_subscription(
            PoseStamped, '/unreachable_goal', self.unreachable_cb, 10)
        self.create_subscription(
            Status, g('status_topic'), self.status_cb, 10)
        self.create_subscription(
            Odometry, '/odometry', self.odom_cb, 10)

        self.lander_start = self.create_client(
            Trigger, '/precision_landing/start')
        self.lander_stop = self.create_client(
            Trigger, '/precision_landing/stop')
        self.mode_client = self.create_client(ModeSwitch, g('mode_service'))

        self.create_service(
            Trigger, '/mission/return_home', self.return_home_cb)
        self.create_service(
            Trigger, '/mission/land_here', self.land_here_cb)

        self.create_timer(1.0 / float(g('tick_rate')), self.tick)
        self.publish_gates()
        self.get_logger().info(
            'Mission manager ready. Waiting for the drone to take off.')

    # --------------------------------------------------
    # Inputs
    # --------------------------------------------------

    def status_cb(self, msg):
        self.status = msg
        self.status_time = _now()

    def odom_cb(self, msg):
        self.odom_z = msg.pose.pose.position.z

    def remaining_cb(self, msg):
        self.remaining = int(msg.data)
        self.remaining_time = _now()

    def unreachable_cb(self, msg):
        self.unreachable.append(
            (msg.pose.position.x, msg.pose.position.y, _now()))
        self.unreachable = self.unreachable[-10:]

    def drone_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x, t.transform.translation.y, yaw)

    def marker_cb(self, msg):
        """Marker position (camera optical frame, from precision_lander)
        -> marker position in the map frame, using the drone's pose.
        Body forward = -ty, body left = -tx (camera looks straight down,
        image top = drone front)."""
        pose = self.drone_pose()
        if pose is None:
            return
        tx, ty = msg.pose.position.x, msg.pose.position.y
        fwd, left = -ty, -tx
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        mx = x + c * fwd - s * left
        my = y + s * fwd + c * left
        self.last_marker_time = _now()
        self.last_marker_map = (mx, my)
        if self.state == 'HOME_ACQUIRE' and self.acquire_start is not None:
            self.samples.append((mx, my))

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def status_fresh(self):
        return (self.status is not None
                and _now() - self.status_time < 3.0)

    def marker_visible(self):
        return _now() - self.last_marker_time < 0.6

    def set_state(self, new, why=''):
        self.get_logger().info(f'[{self.state}] -> [{new}] {why}')
        self.state = new
        self.state_since = _now()
        self.arrive_since = None
        self.publish_gates()

    def publish_gates(self):
        sel, ctl = GATES[self.state]
        m = Bool()
        m.data = sel
        self.selector_gate.publish(m)
        m2 = Bool()
        m2.data = ctl
        self.controller_gate.publish(m2)
        self.last_gate_pub = _now()

    def publish_goal(self, x, y):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        self.last_goal_pub = _now()

    def call_trigger(self, client):
        if client.service_is_ready():
            client.call_async(Trigger.Request())
            return True
        return False

    def request_land_mode(self):
        if not self.mode_client.service_is_ready():
            return False
        req = ModeSwitch.Request()
        req.mode = int(self.p['land_mode'])
        self.mode_client.call_async(req)
        return True

    def warn(self, text, period=5.0):
        if _now() - self.last_warn > period:
            self.last_warn = _now()
            self.get_logger().warning(text)

    # --------------------------------------------------
    # Manual commands
    # --------------------------------------------------

    def return_home_cb(self, req, res):
        if self.state == 'EXPLORE':
            self.set_state('RETURN', 'manual /mission/return_home')
            res.success, res.message = True, 'Returning to the marker'
        elif self.state in ('WAIT_FLYING', 'HOME_ACQUIRE'):
            res.success, res.message = False, 'Not exploring yet'
        else:
            res.success, res.message = False, f'Already in {self.state}'
        return res

    def land_here_cb(self, req, res):
        if self.state in ('LANDING', 'DONE'):
            res.success, res.message = False, f'Already in {self.state}'
            return res
        self.call_trigger(self.lander_stop)
        self.set_state('LANDING', 'manual /mission/land_here')
        self.need_land_mode = not self.request_land_mode()
        res.success, res.message = True, 'LAND requested'
        return res

    # --------------------------------------------------
    # State machine
    # --------------------------------------------------

    def tick(self):
        now = _now()
        if now - self.last_gate_pub > 1.0:
            self.publish_gates()          # keep late subscribers in sync

        if self.state == 'DONE':
            return

        st = self.status if self.status_fresh() else None
        pose = self.drone_pose()
        if st is not None and st.flying:
            self.was_flying = True

        # Vehicle disarmed after having flown: the mission is over
        if (self.was_flying and st is not None and not st.armed
                and self.state != 'WAIT_FLYING'):
            self.set_state('DONE', 'vehicle disarmed')
            return

        handler = getattr(self, f'_do_{self.state.lower()}')
        handler(now, st, pose)

    def _do_wait_flying(self, now, st, pose):
        if st is not None and st.flying and pose is not None:
            self.takeoff_pos = (pose[0], pose[1])
            self.samples = []
            self.acquire_start = None
            self.set_state('HOME_ACQUIRE', 'airborne, measuring marker')

    def _do_home_acquire(self, now, st, pose):
        alt_ok = (self.odom_z is None
                  or self.odom_z >= self.p['home_acquire_alt'])
        if alt_ok and self.acquire_start is None:
            self.acquire_start = now
            self.samples = []
        if self.acquire_start is None:
            return
        enough = len(self.samples) >= self.p['home_samples']
        timeout = now - self.acquire_start > self.p['home_timeout']
        if not (enough or timeout):
            return

        if len(self.samples) >= 3:
            xs = [s[0] for s in self.samples]
            ys = [s[1] for s in self.samples]
            self.home = (statistics.median(xs), statistics.median(ys))
            spread = max(math.hypot(x - self.home[0], y - self.home[1])
                         for x, y in self.samples)
            self.home_source = f'marker ({len(self.samples)} detections, ' \
                               f'spread {spread * 100:.0f} cm)'
        else:
            self.home = self.takeoff_pos
            self.home_source = ('takeoff point (marker not seen, got '
                                f'{len(self.samples)} detections)')
            self.get_logger().warning(
                'Marker not detected at takeoff; using the takeoff point '
                'as home.')
        self.get_logger().info(
            f'Home = ({self.home[0]:.2f}, {self.home[1]:.2f}) from '
            f'{self.home_source}')
        self.explore_start = now
        self.none_since = None
        self.set_state('EXPLORE', 'home recorded')

    def _do_explore(self, now, st, pose):
        elapsed = now - self.explore_start
        limit = self.p['max_explore_time']
        if limit and elapsed > limit:
            self.set_state('RETURN', f'time limit {limit:.0f} s reached')
            return
        if elapsed < self.p['min_explore_time']:
            return

        stale = (self.remaining_time is None
                 or now - self.remaining_time > self.p['frontier_stale'])
        nothing_left = stale or self.remaining == 0
        if not nothing_left:
            self.none_since = None
            return
        if self.none_since is None:
            self.none_since = now
        if now - self.none_since >= self.p['no_frontier_confirm']:
            why = ('selector silent' if stale
                   else 'no frontiers left')
            self.set_state('RETURN', f'exploration finished: {why}')

    def _handoff_if_marker(self):
        """Marker visible and near home => precision landing takes over."""
        if not self.marker_visible() or self.last_marker_map is None:
            return False
        d = math.hypot(self.last_marker_map[0] - self.home[0],
                       self.last_marker_map[1] - self.home[1])
        if d > self.p['handoff_radius']:
            return False
        self.handoff_called = False
        self.set_state(
            'HANDOFF', f'marker in view, {d:.2f} m from recorded home')
        return True

    def _do_return(self, now, st, pose):
        if self._handoff_if_marker():
            return
        if pose is None:
            return
        if now - self.last_goal_pub > 1.0:
            self.publish_goal(*self.home)
        d = math.hypot(pose[0] - self.home[0], pose[1] - self.home[1])
        if d < self.p['arrive_radius']:
            if self.arrive_since is None:
                self.arrive_since = now
            elif now - self.arrive_since > self.p['arrive_dwell']:
                self.search_pts = [
                    (self.home[0] + dx, self.home[1] + dy)
                    for dx, dy in spiral_points(self.p['search_step'],
                                                self.p['search_radius'])]
                self.search_idx = 0
                self.search_idx_since = now
                self.set_state(
                    'SEARCH', 'at home but marker not visible')
        else:
            self.arrive_since = None
        for ux, uy, t in self.unreachable:
            if (now - t < 3.0
                    and math.hypot(ux - self.home[0],
                                   uy - self.home[1]) < 0.5):
                self.warn('Planner cannot reach home; still trying.')

    def _do_search(self, now, st, pose):
        if self._handoff_if_marker():
            return
        if pose is None or not self.search_pts:
            return
        tx, ty = self.search_pts[self.search_idx]
        reached = math.hypot(pose[0] - tx, pose[1] - ty) \
            < self.p['search_point_radius']
        timed_out = now - self.search_idx_since > self.p['search_point_time']
        skipped = any(now - t < 3.0 and math.hypot(ux - tx, uy - ty) < 0.5
                      for ux, uy, t in self.unreachable)
        if reached or timed_out or skipped:
            self.search_idx += 1
            if self.search_idx >= len(self.search_pts):
                self.search_idx = 0
                self.get_logger().warning(
                    'Search pattern finished without seeing the marker; '
                    'repeating. Use /mission/land_here if needed.')
            self.search_idx_since = now
            tx, ty = self.search_pts[self.search_idx]
            self.publish_goal(tx, ty)
            return
        if now - self.last_goal_pub > 1.0:
            self.publish_goal(tx, ty)

    def _do_handoff(self, now, st, pose):
        if not self.handoff_called:
            if self.call_trigger(self.lander_start):
                self.handoff_called = True
            else:
                self.warn('Waiting for /precision_landing/start service...')
            return
        if st is not None and st.mode == int(self.p['land_mode']):
            self.set_state('LANDING', 'LAND mode active, waiting for touchdown')
            return
        if now - self.last_marker_time > self.p['lander_lost_timeout']:
            self.call_trigger(self.lander_stop)
            self.search_pts = [
                (self.home[0] + dx, self.home[1] + dy)
                for dx, dy in spiral_points(self.p['search_step'],
                                            self.p['search_radius'])]
            self.search_idx = 0
            self.search_idx_since = now
            self.set_state(
                'SEARCH',
                f'lander lost the marker for '
                f'{self.p["lander_lost_timeout"]:.0f} s')

    def _do_landing(self, now, st, pose):
        # Normally just waiting for the disarm check in tick(). After a
        # manual land_here, keep retrying the mode switch until it is sent.
        if self.need_land_mode and self.request_land_mode():
            self.need_land_mode = False

    def _do_done(self, now, st, pose):
        pass


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
