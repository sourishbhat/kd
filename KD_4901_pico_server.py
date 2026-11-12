#!/usr/bin/env python3
'''
Action server 'waypoint_navigation' (waypoint_navigation/action/NavToWaypoint).

This node runs the Task 2A PID position controller (hardcoded gains, no tuner GUI) AND the action
server. A goal sets the setpoint; the server streams the drone's current whycode pose as feedback and
succeeds once the drone has stayed inside the tolerance box on X, Y and Z for hold_duration seconds.

Do NOT run pico_controller_PID at the same time: both publish /drone_command.

Axis convention (index -> axis -> command): 0 -> X -> pitch, 1 -> Y -> roll, 2 -> Z -> throttle.
error = setpoint - current. The camera is overhead, so larger Z = LOWER drone (throttle sign -1).
'''

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from waypoint_navigation.action import NavToWaypoint

from swift_msgs.msg import SwiftMsgs
from std_msgs.msg import Header
from whycode_interfaces.msg import MarkerArray
from error_msg.msg import Error
from controller_msg.msg import PIDTune


class WayPointServer(Node):
    def __init__(self):
        super().__init__('waypoint_server')

        # Callback groups: the control timer and the (long running) goal execution must run in parallel
        self.action_callback_group = ReentrantCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.marker_callback_group = MutuallyExclusiveCallbackGroup()

        # ---------------- state ----------------
        self.current_state = [0.0, 0.0, 0.0]      # [x, y, z] whycode units
        self.desired_state = [0.0, 0.0, 13.0]     # set by each goal
        self.marker_received = False
        self.last_marker_time = None
        self.marker_timeout = 0.3                 # s without a marker -> hold neutral
        self.have_goal = False                    # stay neutral until the first goal arrives

        self.yaw = 0.0                            # current yaw, degrees (from whycode marker)
        self.desired_yaw = -0.5                   # target yaw, degrees

        # ---------------- hold detection ----------------
        # Scoring box is +-0.4 on every axis; use a slightly tighter box and a small time margin
        self.tolerance = 0.35
        self.hold_duration = 3.1                  # s (scorer needs 3 s)

        # ---------------- PID (Task 2A tuned values), index [x/pitch, y/roll, z/throttle] ----------------
        self.Kp = [30.0, 30.0, 100.0]
        self.Ki = [0.0, 0.0, 12.0]
        self.Kd = [60.0, 60.0, 80.0]
        self.sign = [1, 1, -1]
        self.neutral = 1500
        self.max_values = [1620, 1620, 2000]
        self.min_values = [1380, 1380, 1000]
        self.max_integral_out = [60.0, 60.0, 60.0]
        self.d_alpha = 0.2
        self.d_filt = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        self.prev_state = None
        self.last_time = None

        # ---------------- yaw P controller ----------------
        self.Kp_yaw = 1.5
        self.yaw_sign = 1                         # flip to -1 if the drone spins away from desired_yaw
        self.yaw_max_offset = 100                 # command stays within 1500 +- this

        self.sample_time = 0.033

        # ---------------- command / publishers ----------------
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

        self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1,
                                 callback_group=self.marker_callback_group)

        # Optional live tuning (off by default so the GUI can never overwrite the hardcoded gains)
        self.declare_parameter('use_tuner_gui', False)
        if self.get_parameter('use_tuner_gui').value:
            self.create_subscription(PIDTune, '/throttle_pid', lambda m: self.set_pid(2, m, 0.03, 0.008, 0.6), 1)
            self.create_subscription(PIDTune, '/pitch_pid', lambda m: self.set_pid(0, m, 0.03, 0.008, 0.6), 1)
            self.create_subscription(PIDTune, '/roll_pid', lambda m: self.set_pid(1, m, 0.03, 0.008, 0.6), 1)

        # ---------------- action server ----------------
        self.action_server = ActionServer(
            self, NavToWaypoint, 'waypoint_navigation', self.execute_callback,
            callback_group=self.action_callback_group)

        self.publish_heartbeat()
        self.create_timer(0.1, self.publish_heartbeat)

        time.sleep(0.5)  # let the heartbeat reach the rosbag before arming
        self.arm()

        self.timer = self.create_timer(self.sample_time, self.pid, callback_group=self.control_callback_group)

    # ------------------------------------------------------------------ helpers
    def publish_heartbeat(self):
        hb = Header()
        hb.stamp = self.get_clock().now().to_msg()
        hb.frame_id = 'pico_controller_active'
        self.heartbeat_pub.publish(hb)

    def disarm(self):
        self.cmd.rc_roll = 1000
        self.cmd.rc_yaw = 1000
        self.cmd.rc_pitch = 1000
        self.cmd.rc_throttle = 1000
        self.cmd.rc_aux4 = 1000
        self.command_pub.publish(self.cmd)

    def arm(self):
        self.disarm()
        self.cmd.rc_roll = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_throttle = 1500
        self.cmd.rc_aux4 = 2000
        self.command_pub.publish(self.cmd)

    def set_pid(self, axis, msg, kp_scale, ki_scale, kd_scale):
        self.Kp[axis] = msg.kp * kp_scale
        self.Ki[axis] = msg.ki * ki_scale
        self.Kd[axis] = msg.kd * kd_scale

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def publish_neutral(self):
        self.cmd.rc_roll = self.neutral
        self.cmd.rc_pitch = self.neutral
        self.cmd.rc_throttle = self.neutral
        self.cmd.rc_yaw = self.neutral
        self.command_pub.publish(self.cmd)

    def reset_controller(self):
        self.error_sum = [0.0, 0.0, 0.0]
        self.d_filt = [0.0, 0.0, 0.0]
        self.prev_state = None
        self.last_time = None

    # ------------------------------------------------------------------ whycode
    def whycode_callback(self, msg):
        if not msg.markers:
            return
        marker = msg.markers[0]
        pos = marker.position.position
        self.current_state = [pos.x, pos.y, pos.z]
        # yaw comes from the marker's own orientation (as in the provided boilerplate), radians -> degrees
        self.yaw = math.degrees(marker.rotation.z)
        self.marker_received = True
        self.last_marker_time = time.monotonic()

    # ------------------------------------------------------------------ controller
    def pid(self):
        if not self.marker_received:
            return

        # Marker lost: hover (neutral) rather than fly on stale data
        if time.monotonic() - self.last_marker_time > self.marker_timeout:
            self.publish_neutral()
            self.reset_controller()
            self.get_logger().warn('Marker lost - holding neutral', throttle_duration_sec=1.0)
            return

        # No goal yet: hover at neutral, don't chase the placeholder setpoint
        if not self.have_goal:
            self.publish_neutral()
            self.reset_controller()
            return

        now = time.monotonic()
        state = list(self.current_state)
        setpoint = list(self.desired_state)

        if self.last_time is None or self.prev_state is None:
            self.last_time = now
            self.prev_state = state
            return
        dt = now - self.last_time
        if dt <= 0.0:
            return
        self.last_time = now

        error = [setpoint[i] - state[i] for i in range(3)]
        out = [0.0, 0.0, 0.0]

        for i in range(3):
            # Integral with anti-windup
            self.error_sum[i] += error[i] * dt
            if self.Ki[i] != 0.0:
                limit = self.max_integral_out[i] / self.Ki[i]
                self.error_sum[i] = self.clamp(self.error_sum[i], -limit, limit)

            # Derivative on the MEASUREMENT (no kick when the setpoint jumps to a new waypoint).
            # d(error)/dt = -d(state)/dt while the setpoint is constant.
            d_raw = -(state[i] - self.prev_state[i]) / dt
            self.d_filt[i] = self.d_alpha * d_raw + (1.0 - self.d_alpha) * self.d_filt[i]

            out[i] = self.Kp[i] * error[i] + self.Ki[i] * self.error_sum[i] + self.Kd[i] * self.d_filt[i]

        self.prev_state = state

        pitch_cmd = self.neutral + self.sign[0] * out[0]
        roll_cmd = self.neutral + self.sign[1] * out[1]
        throttle_cmd = self.neutral + self.sign[2] * out[2]

        # Yaw: P controller on the wrapped angle error
        yaw_error = (self.desired_yaw - self.yaw + 180.0) % 360.0 - 180.0
        yaw_offset = self.clamp(self.yaw_sign * self.Kp_yaw * yaw_error, -self.yaw_max_offset, self.yaw_max_offset)

        self.cmd.rc_pitch = int(self.clamp(pitch_cmd, self.min_values[0], self.max_values[0]))
        self.cmd.rc_roll = int(self.clamp(roll_cmd, self.min_values[1], self.max_values[1]))
        self.cmd.rc_throttle = int(self.clamp(throttle_cmd, self.min_values[2], self.max_values[2]))
        self.cmd.rc_yaw = int(self.neutral + yaw_offset)
        self.command_pub.publish(self.cmd)

        pos_error = Error()
        pos_error.pitch_error = float(error[0])
        pos_error.roll_error = float(error[1])
        pos_error.throttle_error = float(error[2])
        self.pos_error_pub.publish(pos_error)

    # ------------------------------------------------------------------ action
    def in_box(self):
        return all(abs(self.desired_state[i] - self.current_state[i]) <= self.tolerance for i in range(3))

    def execute_callback(self, goal_handle):
        wp = goal_handle.request.waypoint.position
        self.desired_state = [wp.x, wp.y, wp.z]
        self.have_goal = True
        self.get_logger().info('New waypoint set: (%.2f, %.2f, %.2f)' % (wp.x, wp.y, wp.z))

        start = time.monotonic()
        hold_start = None
        feedback_msg = NavToWaypoint.Feedback()

        while rclpy.ok():
            # Feedback: the drone's current whycode pose
            feedback_msg.current_waypoint.header.stamp = self.get_clock().now().to_msg()
            feedback_msg.current_waypoint.header.frame_id = 'whycode'
            feedback_msg.current_waypoint.pose.position.x = self.current_state[0]
            feedback_msg.current_waypoint.pose.position.y = self.current_state[1]
            feedback_msg.current_waypoint.pose.position.z = self.current_state[2]
            goal_handle.publish_feedback(feedback_msg)

            now = time.monotonic()
            if self.in_box():
                if hold_start is None:
                    hold_start = now
                elif now - hold_start >= self.hold_duration:
                    break
            else:
                hold_start = None  # left the box: the 3 s must be continuous

            time.sleep(0.02)
        else:
            goal_handle.abort()
            return NavToWaypoint.Result()

        goal_handle.succeed()
        result = NavToWaypoint.Result()
        result.hov_time = int(round(time.monotonic() - start))  # total time to stabilise at this point
        self.get_logger().info('Waypoint reached and held (%d s)' % result.hov_time)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = WayPointServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
