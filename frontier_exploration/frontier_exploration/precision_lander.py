#!/usr/bin/env python3
"""
ArUco precision landing for ArduPilot (DDS) + Gazebo.

Looks for an ArUco marker with the downward camera, estimates where the
marker is relative to the drone, steers over it with body-frame velocity
commands on /ap/v1/cmd_vel, descends once centred, and finally switches
ArduPilot to LAND.

Start it with:   ros2 service call /precision_landing/start std_srvs/srv/Trigger
(or set the parameter auto_start:=true).

IMPORTANT: this node and frontier_controller both publish /ap/v1/cmd_vel.
Stop (or pause) the frontier controller before starting the landing,
otherwise the two fight over the setpoint.

Camera convention (see model.sdf): camera looks straight down, image top =
drone front, image right = drone right. In the OpenCV optical frame
(x right, y down, z out of the lens) that gives, for a marker at
(tx, ty, tz) relative to the camera:
    marker forward of the drone  =  -ty
    marker left of the drone     =  -tx
    height above the marker      =   tz
"""

import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_srvs.srv import Trigger
from ardupilot_msgs.msg import Status
from ardupilot_msgs.srv import ModeSwitch


# ----------------------------------------------------------------------
# Pure helpers (no ROS) -- these are unit-tested separately
# ----------------------------------------------------------------------

def image_to_gray(msg):
    """sensor_msgs/Image -> uint8 gray, without cv_bridge."""
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding
    if enc in ('rgb8', 'bgr8'):
        img = buf.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        code = cv2.COLOR_RGB2GRAY if enc == 'rgb8' else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)
    if enc in ('rgba8', 'bgra8'):
        img = buf.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)
        code = cv2.COLOR_RGBA2GRAY if enc == 'rgba8' else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(img, code)
    if enc == 'mono8':
        return buf.reshape(h, msg.step)[:, :w].copy()
    raise ValueError(f'unsupported image encoding: {enc}')


def make_detector(dict_name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    # While descending the marker fills most of the frame and its white
    # quiet zone reaches the image edge. With the default border filter the
    # detector then drops the marker (found in testing at ~1 m height).
    params.minDistanceToBorder = 0
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, params)


def estimate_marker(gray, detector, marker_id, marker_size, K, D):
    """Find `marker_id` and return (tvec(3,), corners(4,2)) or None.
    tvec is the marker centre in the camera optical frame, in metres."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten()):
        if int(i) != marker_id:
            continue
        c = c.reshape(4, 2).astype(np.float64)
        # The camera looks straight down at a flat marker, so we do not need
        # a full pose solve. solvePnP (IPPE) has a tilt ambiguity for a flat
        # square seen nearly head-on and was measured 3-4 cm off at 0.25 m
        # offset / 1 m height. Instead: undistort the corners to normalised
        # coordinates (X/Z, Y/Z), get the depth from the apparent edge
        # length (edge_norm = size / Z) and place the centre from that.
        n = cv2.undistortPoints(c.reshape(-1, 1, 2), K, D).reshape(4, 2)
        edge = np.mean([np.linalg.norm(n[i] - n[(i + 1) % 4])
                        for i in range(4)])
        if edge < 1e-6:
            return None
        z = marker_size / edge
        cx_n, cy_n = n.mean(axis=0)
        return np.array([cx_n * z, cy_n * z, z]), c
    return None


def marker_error_body(tvec):
    """Marker position relative to the drone in body FLU axes, plus the
    height above the marker. Assumes the camera sits over the body centre."""
    tx, ty, tz = float(tvec[0]), float(tvec[1]), float(tvec[2])
    return -ty, -tx, tz       # forward, left, height


def servo_command(ex, ey, height, kp, max_speed, descent_speed,
                  align_tol_min, align_tol_frac):
    """Body-frame (vx, vy, vz) in FLU. Descends only when centred; the
    centring tolerance grows with height so it is not impossibly strict
    from far away and tightens as the drone gets close."""
    vx, vy = kp * ex, kp * ey
    speed = math.hypot(vx, vy)
    if speed > max_speed:
        vx, vy = vx * max_speed / speed, vy * max_speed / speed
    tol = max(align_tol_min, align_tol_frac * height)
    vz = -descent_speed if math.hypot(ex, ey) < tol else 0.0
    return vx, vy, vz


# ----------------------------------------------------------------------
# Node
# ----------------------------------------------------------------------

class PrecisionLander(Node):

    def __init__(self):
        super().__init__('precision_lander')

        dp = self.declare_parameter
        dp('image_topic', '/downward_camera/image_raw')
        dp('camera_info_topic', '/downward_camera/camera_info')
        dp('cmd_vel_topic', '/ap/v1/cmd_vel')
        dp('status_topic', '/ap/v1/status')
        dp('mode_service', '/ap/v1/mode_switch')

        dp('dictionary', 'DICT_4X4_50')
        dp('marker_id', 0)
        dp('marker_size', 0.50)        # black square only, metres

        dp('kp', 0.8)
        dp('max_speed', 0.35)
        dp('descent_speed', 0.20)
        dp('align_tol_min', 0.08)      # m
        dp('align_tol_frac', 0.10)     # of current height
        dp('land_height', 0.40)        # switch to LAND below this height
        dp('land_mode', 9)             # Copter LAND
        dp('lost_hold_time', 1.0)      # s: hold still after losing marker
        dp('lost_climb_time', 4.0)     # s: then climb to re-acquire
        dp('climb_speed', 0.15)
        dp('filter_alpha', 0.5)        # EMA on the marker position
        dp('control_rate', 20.0)
        dp('auto_start', False)
        dp('dry_run', False)

        # Same sign convention as frontier_controller: the body-frame
        # command needed y flipped on this setup.
        dp('flip_x', False)
        dp('flip_y', True)
        dp('debug_image', True)

        g = lambda n: self.get_parameter(n).value
        self.marker_id = int(g('marker_id'))
        self.marker_size = float(g('marker_size'))
        self.kp = float(g('kp'))
        self.max_speed = float(g('max_speed'))
        self.descent_speed = float(g('descent_speed'))
        self.align_tol_min = float(g('align_tol_min'))
        self.align_tol_frac = float(g('align_tol_frac'))
        self.land_height = float(g('land_height'))
        self.land_mode = int(g('land_mode'))
        self.lost_hold = float(g('lost_hold_time'))
        self.lost_climb = float(g('lost_climb_time'))
        self.climb_speed = float(g('climb_speed'))
        self.alpha = float(g('filter_alpha'))
        self.dry_run = bool(g('dry_run'))
        self.flip_x = -1.0 if g('flip_x') else 1.0
        self.flip_y = -1.0 if g('flip_y') else 1.0
        self.want_debug_image = bool(g('debug_image'))

        self.detector = make_detector(str(g('dictionary')))
        self.K = None
        self.D = None

        self.active = bool(g('auto_start'))
        self.landing_sent = False
        self.filtered = None          # (ex, ey, height)
        self.last_seen = 0.0
        self.last_height = None
        self.status = None
        self.status_time = 0.0
        self.last_log = 0.0

        self.create_subscription(
            CameraInfo, g('camera_info_topic'), self.info_cb,
            qos_profile_sensor_data)
        self.create_subscription(
            Image, g('image_topic'), self.image_cb, qos_profile_sensor_data)
        self.create_subscription(
            Status, g('status_topic'), self.status_cb, 10)

        self.cmd_pub = self.create_publisher(
            TwistStamped, g('cmd_vel_topic'), 10)
        self.pose_pub = self.create_publisher(
            PoseStamped, '/precision_landing/marker_pose', 10)
        self.debug_pub = self.create_publisher(
            Image, '/precision_landing/debug_image', 10)

        self.mode_client = self.create_client(ModeSwitch, g('mode_service'))
        self.create_service(Trigger, '/precision_landing/start',
                            self.start_cb)
        self.create_service(Trigger, '/precision_landing/stop',
                            self.stop_cb)

        self.create_timer(1.0 / float(g('control_rate')), self.control_loop)
        self.get_logger().info(
            f'Precision lander ready (id {self.marker_id}, '
            f'{self.marker_size:.2f} m). active={self.active} '
            f'dry_run={self.dry_run}')

    # ---------------- callbacks ----------------

    def info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            d = np.array(msg.d, dtype=np.float64)
            self.D = d if d.size else np.zeros(5)
            self.get_logger().info(
                f'Camera intrinsics: fx={self.K[0, 0]:.1f} '
                f'fy={self.K[1, 1]:.1f} cx={self.K[0, 2]:.1f} '
                f'cy={self.K[1, 2]:.1f}')

    def status_cb(self, msg):
        self.status = msg
        self.status_time = time.monotonic()

    def start_cb(self, req, res):
        self.active = True
        self.landing_sent = False
        res.success = True
        res.message = 'Precision landing started'
        self.get_logger().info(res.message)
        return res

    def stop_cb(self, req, res):
        self.active = False
        res.success = True
        res.message = 'Precision landing stopped'
        self.get_logger().info(res.message)
        return res

    def image_cb(self, msg):
        if self.K is None:
            return
        try:
            gray = image_to_gray(msg)
        except ValueError as ex:
            self.get_logger().error(str(ex), throttle_duration_sec=5.0)
            return

        found = estimate_marker(gray, self.detector, self.marker_id,
                                self.marker_size, self.K, self.D)
        if found is None:
            return

        tvec, corners = found
        ex, ey, h = marker_error_body(tvec)
        if self.filtered is None or time.monotonic() - self.last_seen > 1.0:
            self.filtered = (ex, ey, h)
        else:
            a = self.alpha
            fx, fy, fh = self.filtered
            self.filtered = (a * ex + (1 - a) * fx,
                             a * ey + (1 - a) * fy,
                             a * h + (1 - a) * fh)
        self.last_seen = time.monotonic()
        self.last_height = self.filtered[2]

        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = 'camera_optical_frame'
        pose.pose.position.x = float(tvec[0])
        pose.pose.position.y = float(tvec[1])
        pose.pose.position.z = float(tvec[2])
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)

        if self.want_debug_image and self.debug_pub.get_subscription_count():
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, [corners.astype(np.int32)], True,
                          (0, 255, 0), 2)
            cv2.circle(vis, (int(self.K[0, 2]), int(self.K[1, 2])), 6,
                       (0, 0, 255), 2)
            out = Image()
            out.header = msg.header
            out.height, out.width = vis.shape[:2]
            out.encoding = 'bgr8'
            out.step = out.width * 3
            out.data = vis.tobytes()
            self.debug_pub.publish(out)

    # ---------------- control ----------------

    def send(self, vx, vy, vz):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(vx) * self.flip_x
        msg.twist.linear.y = float(vy) * self.flip_y
        msg.twist.linear.z = float(vz)
        if not self.dry_run:
            self.cmd_pub.publish(msg)

    def request_land(self):
        if self.landing_sent:
            return
        if not self.mode_client.service_is_ready():
            self.get_logger().warning(
                'Mode service not ready, retrying...',
                throttle_duration_sec=2.0)
            return
        req = ModeSwitch.Request()
        req.mode = self.land_mode
        if not self.dry_run:
            self.mode_client.call_async(req)
        self.landing_sent = True
        self.get_logger().info('Low enough over the marker: switching to LAND')

    def control_loop(self):
        if not self.active or self.landing_sent:
            return

        now = time.monotonic()
        status_ok = (self.status is not None
                     and now - self.status_time < 3.0
                     and self.status.armed and self.status.flying)
        if not status_ok:
            return

        since_seen = now - self.last_seen if self.last_seen else math.inf

        if since_seen < 0.3 and self.filtered is not None:
            ex, ey, h = self.filtered
            if h <= self.land_height:
                self.request_land()
                return
            vx, vy, vz = servo_command(
                ex, ey, h, self.kp, self.max_speed, self.descent_speed,
                self.align_tol_min, self.align_tol_frac)
            self.send(vx, vy, vz)
            state = 'DESCEND' if vz < 0 else 'ALIGN'
            if now - self.last_log > 0.5:
                self.last_log = now
                self.get_logger().info(
                    f'[{state}] marker fwd={ex:+.2f} left={ey:+.2f} '
                    f'height={h:.2f} m -> v=({vx:+.2f},{vy:+.2f},{vz:+.2f})')
            return

        # Marker not visible
        if (self.last_height is not None
                and self.last_height <= self.land_height + 0.15):
            # too close for the marker to fit in view: it is right below us
            self.request_land()
            return

        if since_seen < self.lost_hold:
            self.send(0.0, 0.0, 0.0)
        elif since_seen < self.lost_hold + self.lost_climb:
            self.send(0.0, 0.0, self.climb_speed)   # climb to widen the view
            msg = 'Marker lost: climbing to re-acquire'
        else:
            self.send(0.0, 0.0, 0.0)
            msg = 'Marker not found: holding position'
        if since_seen >= self.lost_hold and now - self.last_log > 2.0:
            self.last_log = now
            self.get_logger().warning(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLander()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
