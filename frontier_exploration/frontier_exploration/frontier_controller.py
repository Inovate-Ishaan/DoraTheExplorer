#!/usr/bin/env python3
"""
Frontier controller for ArduPilot (DDS) + Gazebo.

Startup state machine (fully autonomous):
    WAIT_STATUS -> GUIDED -> ARM -> TAKEOFF -> CLIMB -> EXPLORE

In EXPLORE it follows /planned_path (map frame) with a lookahead
P-controller and sends velocity commands to ArduPilot.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from ardupilot_msgs.msg import Status
from ardupilot_msgs.srv import ArmMotors, ModeSwitch, Takeoff

import tf2_ros


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


class FrontierController(Node):

    def __init__(self):
        super().__init__('frontier_controller')

        # ---------------- Parameters ----------------
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('max_speed', 0.95)
        self.declare_parameter('kp', 2) #2
        self.declare_parameter('lookahead', 0.25)
        self.declare_parameter('goal_tolerance', 0.18)
        self.declare_parameter('path_timeout', 8.0)
        self.declare_parameter('dry_run', False)

        # Autonomous startup
        self.declare_parameter('auto_start', True)
        self.declare_parameter('takeoff_alt', 2.2)
        self.declare_parameter('takeoff_timeout', 30.0)
        self.declare_parameter('guided_mode', 4)        # Copter GUIDED
        self.declare_parameter('service_retry_period', 2.0)

        # Altitude hold / heading
        self.declare_parameter('kp_z', 1.0)
        self.declare_parameter('max_vz', 0.4)
        self.declare_parameter('face_motion', False)
        self.declare_parameter('kp_yaw', 1.0)
        self.declare_parameter('max_yaw_rate', 0.8)

        # Topics / services / frames (override if your namespace differs)
        self.declare_parameter('cmd_vel_topic', '/ap/v1/cmd_vel')
        self.declare_parameter('status_topic', '/ap/v1/status')
        self.declare_parameter('arm_service', '/ap/v1/arm_motors')
        self.declare_parameter('mode_service', '/ap/v1/mode_switch')
        self.declare_parameter('takeoff_service',
                               '/ap/v1/experimental/takeoff')
        # 'base_link' = body (FLU) frame; robust against map/EKF yaw offset
        self.declare_parameter('cmd_frame', 'base_link')
        # Axis sign correction, if ArduPilot's frame convention differs
        self.declare_parameter('flip_x', False)
        self.declare_parameter('flip_y', True)
        self.declare_parameter('debug', True)

        gp = self.get_parameter
        self.control_rate = gp('control_rate').value
        self.max_speed = gp('max_speed').value
        self.kp = gp('kp').value
        self.lookahead = gp('lookahead').value
        self.goal_tolerance = gp('goal_tolerance').value
        self.path_timeout = gp('path_timeout').value
        self.dry_run = gp('dry_run').value
        self.auto_start = gp('auto_start').value
        self.takeoff_alt = gp('takeoff_alt').value
        self.takeoff_timeout = gp('takeoff_timeout').value
        self.guided_mode = int(gp('guided_mode').value)
        self.retry_period = gp('service_retry_period').value
        self.kp_z = gp('kp_z').value
        self.max_vz = gp('max_vz').value
        self.face_motion = gp('face_motion').value
        self.kp_yaw = gp('kp_yaw').value
        self.max_yaw_rate = gp('max_yaw_rate').value
        self.cmd_frame = gp('cmd_frame').value
        self.flip_x = -1.0 if gp('flip_x').value else 1.0
        self.flip_y = -1.0 if gp('flip_y').value else 1.0
        self.debug = gp('debug').value

        # ---------------- State ----------------
        self.path = None
        self.odom = None
        self.status = None

        self.path_received_time = 0.0
        self.odom_received_time = 0.0
        self.status_received_time = 0.0

        self.yaw = 0.0  # yaw of base_link in map
        # Mission gate: when False the controller stays silent (takeoff
        # still runs). Defaults to True so it works without a manager.
        self.drive_enabled = True

        self.state = 'WAIT_STATUS' if self.auto_start else 'EXPLORE'
        self.pending = None          # (name, future, start_time)
        self.last_call_time = 0.0
        self.climb_start = 0.0

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self)

        # ---------------- Subscribers ----------------
        self.create_subscription(
            Path, '/planned_path', self.path_callback, 10)
        self.create_subscription(
            Odometry, '/odometry', self.odom_callback, 10)
        self.create_subscription(
            Status, gp('status_topic').value, self.status_callback, 10)
        self.create_subscription(
            Bool, '/mission/controller_enable', self.gate_callback,
            QoSProfile(depth=1,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))

        # ---------------- Publishers ----------------
        self.cmd_pub = self.create_publisher(
            TwistStamped, gp('cmd_vel_topic').value, 10)
        self.preview_pub = self.create_publisher(
            TwistStamped, '/frontier_controller/cmd_vel_preview', 10)

        # ---------------- Service clients ----------------
        # Clients are created lazily by type-based discovery, so the
        # exact service names/namespaces don't matter.
        self.arm_client = None
        self.mode_client = None
        self.takeoff_client = None
        self.last_discovery = 0.0
        self.svc_hint = {
            'arm': gp('arm_service').value,
            'mode': gp('mode_service').value,
            'takeoff': gp('takeoff_service').value,
        }

        self.timer = self.create_timer(
            1.0 / self.control_rate, self.control_loop)

        self.get_logger().info('Frontier Controller initialized.')
        self.get_logger().info(
            f'dry_run={self.dry_run} auto_start={self.auto_start} '
            f'takeoff_alt={self.takeoff_alt} m')

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def path_callback(self, msg):
        if not msg.poses:
            self.path = None
            return
        if msg.header.frame_id != 'map':
            self.get_logger().warning(
                f'Expected map frame, received {msg.header.frame_id}')
            return
        self.path = msg
        self.path_received_time = time.monotonic()

    def odom_callback(self, msg):
        self.odom = msg
        self.odom_received_time = time.monotonic()

    def gate_callback(self, msg):
        self.drive_enabled = bool(msg.data)

    def status_callback(self, msg):
        self.status = msg
        self.status_received_time = time.monotonic()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def status_fresh(self):
        return (self.status is not None
                and time.monotonic() - self.status_received_time < 3.0)

    def altitude(self):
        if self.odom is None:
            return None
        return self.odom.pose.pose.position.z

    def get_map_pose(self):
        """Drone (x, y, yaw) in the map frame via TF map->base_link."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return (t.transform.translation.x,
                t.transform.translation.y, yaw)

    # --------------------------------------------------
    # Autonomous startup state machine
    # --------------------------------------------------

    def discover_services(self):
        """Find ArduPilot services by their type, whatever their name."""
        if (self.arm_client and self.mode_client
                and self.takeoff_client):
            return
        now = time.monotonic()
        if now - self.last_discovery < 2.0:
            return
        self.last_discovery = now

        wanted = {
            'ardupilot_msgs/srv/ArmMotors': ('arm_client', ArmMotors),
            'ardupilot_msgs/srv/ModeSwitch': ('mode_client', ModeSwitch),
            'ardupilot_msgs/srv/Takeoff': ('takeoff_client', Takeoff),
        }
        for name, types in self.get_service_names_and_types():
            for t in types:
                if t in wanted:
                    attr, srv_type = wanted[t]
                    if getattr(self, attr) is None:
                        setattr(self, attr,
                                self.create_client(srv_type, name))
                        self.get_logger().info(
                            f'Found {t} service at {name}')

        missing = [a for a in ('arm_client', 'mode_client',
                               'takeoff_client')
                   if getattr(self, a) is None]
        if missing:
            ap = [n for n, _ in self.get_service_names_and_types()
                  if 'ap' in n.lower() or 'arm' in n.lower()]
            self.get_logger().warning(
                f'Still missing {missing}. ArduPilot-ish services '
                f'visible: {ap}. Is the DDS agent / SITL running with '
                f'DDS enabled (DDS_ENABLE=1)?',
                throttle_duration_sec=10.0)

    def _call(self, name, client, req):
        if client is None or not client.service_is_ready():
            self.get_logger().info(
                f'Waiting for {name} service...',
                throttle_duration_sec=5.0)
            self.last_call_time = time.monotonic()
            return
        self.pending = (name, client.call_async(req), time.monotonic())
        self.last_call_time = time.monotonic()

    def _check_pending(self):
        """Returns (done, success)."""
        name, fut, started = self.pending
        if not fut.done():
            if time.monotonic() - started > 5.0:
                self.get_logger().warning(f'{name}: service call timed out')
                self.pending = None
                return True, False
            return False, False

        self.pending = None
        try:
            res = fut.result()
        except Exception as ex:
            self.get_logger().warning(f'{name}: call failed: {ex}')
            return True, False

        ok = bool(getattr(res, 'result', getattr(res, 'status', False)))
        self.get_logger().info(f'{name}: {"accepted" if ok else "rejected"}')
        return True, ok

    def startup_step(self):
        """Advance startup. Returns True once EXPLORE is reached."""
        if self.state == 'EXPLORE':
            return True
        if self.state == 'ABORT':
            return False

        now = time.monotonic()

        self.discover_services()

        if not self.status_fresh():
            self.get_logger().info(
                'Waiting for /ap status...', throttle_duration_sec=5.0)
            return False

        st = self.status

        # Handle an outstanding service call
        if self.pending is not None:
            done, ok = self._check_pending()
            if done and ok and self.state == 'TAKEOFF':
                self.state = 'CLIMB'
                self.climb_start = now
            return False

        # Hardening: if we got disarmed mid-startup, restart sequence
        if self.state in ('TAKEOFF', 'CLIMB') and not st.armed:
            self.get_logger().warning('Disarmed during startup; retrying')
            self.state = 'GUIDED'

        if self.state == 'WAIT_STATUS':
            self.state = 'GUIDED'

        if now - self.last_call_time < self.retry_period:
            return False

        if self.state == 'GUIDED':
            if st.mode == self.guided_mode:
                self.state = 'ARM'
            else:
                req = ModeSwitch.Request()
                req.mode = self.guided_mode
                self._call('mode_switch(GUIDED)', self.mode_client, req)

        elif self.state == 'ARM':
            if st.mode != self.guided_mode:
                self.state = 'GUIDED'
            elif st.armed:
                self.state = 'TAKEOFF'
            else:
                req = ArmMotors.Request()
                req.arm = True
                self._call('arm', self.arm_client, req)

        elif self.state == 'TAKEOFF':
            if st.flying:
                self.state = 'CLIMB'
                self.climb_start = now
            else:
                req = Takeoff.Request()
                req.alt = float(self.takeoff_alt)
                self._call('takeoff', self.takeoff_client, req)

        elif self.state == 'CLIMB':
            z = self.altitude()
            reached = (z is not None
                       and z >= self.takeoff_alt - 0.15)
            timed_out = now - self.climb_start > self.takeoff_timeout
            if reached and st.flying:
                self.get_logger().info(
                    f'Takeoff complete (z={z:.2f} m). Exploring.')
                self.state = 'EXPLORE'
            elif timed_out:
                if st.flying:
                    self.get_logger().warning(
                        'Climb timeout but flying; starting exploration')
                    self.state = 'EXPLORE'
                else:
                    self.get_logger().warning(
                        'Takeoff timed out; retrying')
                    self.state = 'TAKEOFF'

        return self.state == 'EXPLORE'

    # --------------------------------------------------
    # Path following
    # --------------------------------------------------

    def get_target_waypoint(self, x, y):
        points = self.path.poses

        nearest_idx = min(
            range(len(points)),
            key=lambda i: (
                (points[i].pose.position.x - x) ** 2 +
                (points[i].pose.position.y - y) ** 2))

        for i in range(nearest_idx, len(points)):
            px = points[i].pose.position.x
            py = points[i].pose.position.y
            if math.hypot(px - x, py - y) >= self.lookahead:
                return px, py

        final = points[-1].pose.position
        return final.x, final.y

    def calculate_velocity(self, x, y, target_x, target_y):
        vx = self.kp * (target_x - x)
        vy = self.kp * (target_y - y)

        speed = math.hypot(vx, vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
        return vx, vy

    # --------------------------------------------------
    # Publish command (vx, vy given in MAP frame)
    # --------------------------------------------------

    def publish_command(self, vx, vy):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        vz = 0.0
        z = self.altitude()
        if z is not None and self.state == 'EXPLORE':
            vz = max(-self.max_vz, min(
                self.max_vz, self.kp_z * (self.takeoff_alt - z)))

        wz = 0.0
        if self.face_motion and math.hypot(vx, vy) > 0.1:
            err = wrap_pi(math.atan2(vy, vx) - self.yaw)
            wz = max(-self.max_yaw_rate,
                     min(self.max_yaw_rate, self.kp_yaw * err))

        if self.cmd_frame == 'base_link':
            # rotate map-frame velocity into body (FLU) frame
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            bx = c * vx + s * vy
            by = -s * vx + c * vy
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x = float(bx) * self.flip_x
            msg.twist.linear.y = float(by) * self.flip_y
        else:
            msg.header.frame_id = self.cmd_frame
            msg.twist.linear.x = float(vx) * self.flip_x
            msg.twist.linear.y = float(vy) * self.flip_y

        if (self.debug and (abs(vx) > 1e-3 or abs(vy) > 1e-3)):
            self.get_logger().info(
                f'[dbg] yaw={math.degrees(self.yaw):.0f}deg '
                f'v_map=({vx:.2f},{vy:.2f}) '
                f'cmd[{msg.header.frame_id}]=('
                f'{msg.twist.linear.x:.2f},{msg.twist.linear.y:.2f})',
                throttle_duration_sec=0.5)

        msg.twist.linear.z = float(vz)
        msg.twist.angular.z = float(wz)

        self.preview_pub.publish(msg)

        if self.dry_run:
            return

        safe_to_control = (
            self.status_fresh()
            and self.status.armed
            and self.status.flying
            and self.status.external_control
            and len(self.status.failsafe) == 0)

        if safe_to_control:
            self.cmd_pub.publish(msg)
        else:
            stop = TwistStamped()
            stop.header.stamp = msg.header.stamp
            stop.header.frame_id = msg.header.frame_id
            self.cmd_pub.publish(stop)

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    def control_loop(self):
        now = time.monotonic()

        # Startup phase (arm / takeoff). Not controlled by cmd_vel.
        if not self.dry_run and self.auto_start:
            if not self.startup_step():
                return

        # Abort if we lose the vehicle after exploring has started
        if (self.state == 'EXPLORE' and self.auto_start
                and self.status_fresh() and not self.status.armed
                and not self.dry_run):
            self.get_logger().error(
                'Vehicle disarmed during exploration. Halting.',
                throttle_duration_sec=5.0)
            self.state = 'ABORT'
            return

        # Mission manager has taken over (or is measuring): publish nothing
        if not self.drive_enabled:
            return

        pose = self.get_map_pose()
        if pose is not None:
            self.yaw = pose[2]

        # Validate odometry
        if self.odom is None or now - self.odom_received_time > 0.5:
            self.publish_command(0.0, 0.0)
            return

        # Validate path
        if (self.path is None
                or now - self.path_received_time > self.path_timeout):
            self.publish_command(0.0, 0.0)
            return

        if pose is None:
            self.publish_command(0.0, 0.0)
            return

        x, y, _ = pose

        final = self.path.poses[-1].pose.position
        if math.hypot(final.x - x, final.y - y) < self.goal_tolerance:
            self.publish_command(0.0, 0.0)
            self.get_logger().info(
                'Frontier reached!', throttle_duration_sec=5.0)
            return

        tx, ty = self.get_target_waypoint(x, y)
        vx, vy = self.calculate_velocity(x, y, tx, ty)
        if self.debug:
            self.get_logger().info(
                f'[dbg] pos=({x:.2f},{y:.2f}) target=({tx:.2f},{ty:.2f}) '
                f'goal=({final.x:.2f},{final.y:.2f})',
                throttle_duration_sec=0.5)
        self.publish_command(vx, vy)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.publish_command(0.0, 0.0)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
