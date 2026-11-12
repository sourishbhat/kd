#!/usr/bin/env python3
'''
Task 3A action server 'waypoint_navigation' (waypoint_navigation/action/NavToWaypoint).

Runs the Task 2A PID controller (hardcoded gains) together with the action server.
Goal encoding (the action definition is unchanged):
    hover = False                          -> pass-through point: succeed as soon as the drone is close
    hover = True,  orientation.w == 1.0    -> survivor: Visit (Z=13, 1 s) then Save (Z=14, 2 s)
    hover = True,  orientation.w == 2.0    -> F6 home: hover (Z=13, 1 s), then descend and land

A cancel request is accepted: the goal stops and the drone holds where it is until the next goal.
Every hold is inside a SPHERE around the hold point, and restarts from zero if the drone leaves the sphere
or the marker is lost.

Do NOT run pico_controller_PID at the same time: both publish /drone_command.

Axis convention (index -> axis -> command): 0 -> X -> pitch, 1 -> Y -> roll, 2 -> Z -> throttle.
error = setpoint - current. The camera is overhead: larger Z = LOWER drone (throttle sign -1).
'''

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
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

        self.action_callback_group = ReentrantCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.marker_callback_group = MutuallyExclusiveCallbackGroup()

        # ---------------- state ----------------
        self.current_state = [0.0, 0.0, 0.0]      # [x, y, z] whycode units
        self.desired_state = [0.0, 0.0, 13.0]
        self.marker_received = False
        self.last_marker_time = None
        self.marker_timeout = 0.3                 # s without a marker -> hold neutral / reset holds
        self.have_goal = False                    # stay neutral until the first goal arrives
        self.landed = False                       # True after touchdown: motors disarmed

        self.yaw = 0.0
        self.desired_yaw = -0.5

        # Ground height (Z of the marker while the drone sits on F6 before take-off). 0.0 = measure at start.
        self.declare_parameter('ground_z', 0.0)
        self.ground_z = float(self.get_parameter('ground_z').value) or None

        # ---------------- mission geometry ----------------
        self.fly_z = 13.0
        self.save_z = 14.0
        # The scorer uses a sphere of radius 0.4 around each hold point: use a tighter one plus a time margin
        self.sphere_radius = 0.32
        self.visit_hold = 1.1                     # s (scorer needs 1)
        self.save_hold = 2.1                      # s (scorer needs 2)
        self.home_hold = 1.1                      # s (scorer needs 1)
        self.pass_radius = 0.5                    # pass-through points: just get close, no hold
        self.descent_rate = 0.5                   # whycode units per second while landing
        self.touchdown_margin = 0.2               # disarm when this close to the ground

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
        self.yaw_max_offset = 100

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

        # ---------------- action server (cancel enabled) ----------------
        self.action_server = ActionServer(
            self, NavToWaypoint, 'waypoint_navigation', self.execute_callback,
            cancel_callback=self.cancel_callback,
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

    def marker_fresh(self):
        return self.last_marker_time is not None and time.monotonic() - self.last_marker_time <= self.marker_timeout

    # ------------------------------------------------------------------ whycode
    def whycode_callback(self, msg):
        if not msg.markers:
            return
        marker = msg.markers[0]
        pos = marker.position.position
        self.current_state = [pos.x, pos.y, pos.z]
        self.yaw = math.degrees(marker.rotation.z)
        if self.ground_z is None:
            self.ground_z = pos.z  # the drone is sitting on F6 when the server starts
            self.get_logger().info('Ground level measured: Z = %.2f' % self.ground_z)
        self.marker_received = True
        self.last_marker_time = time.monotonic()

    # ------------------------------------------------------------------ controller
    def pid(self):
        if self.landed:  # touched down: keep the motors off
            self.disarm()
            return
        if not self.marker_received:
            return

        if not self.marker_fresh():
            self.publish_neutral()
            self.reset_controller()
            self.get_logger().warn('Marker lost - holding neutral', throttle_duration_sec=1.0)
            return

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
            self.error_sum[i] += error[i] * dt
            if self.Ki[i] != 0.0:
                limit = self.max_integral_out[i] / self.Ki[i]
                self.error_sum[i] = self.clamp(self.error_sum[i], -limit, limit)

            # Derivative on the measurement: no kick when the setpoint jumps
            d_raw = -(state[i] - self.prev_state[i]) / dt
            self.d_filt[i] = self.d_alpha * d_raw + (1.0 - self.d_alpha) * self.d_filt[i]

            out[i] = self.Kp[i] * error[i] + self.Ki[i] * self.error_sum[i] + self.Kd[i] * self.d_filt[i]

        self.prev_state = state

        pitch_cmd = self.neutral + self.sign[0] * out[0]
        roll_cmd = self.neutral + self.sign[1] * out[1]
        throttle_cmd = self.neutral + self.sign[2] * out[2]

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

    # ------------------------------------------------------------------ action helpers
    def cancel_callback(self, goal_handle):
        self.get_logger().info('Cancel requested')
        return CancelResponse.ACCEPT

    def send_feedback(self, goal_handle):
        fb = NavToWaypoint.Feedback()
        fb.current_waypoint.header.stamp = self.get_clock().now().to_msg()
        fb.current_waypoint.header.frame_id = 'whycode'
        fb.current_waypoint.pose.position.x = self.current_state[0]
        fb.current_waypoint.pose.position.y = self.current_state[1]
        fb.current_waypoint.pose.position.z = self.current_state[2]
        goal_handle.publish_feedback(fb)

    def hold_in_sphere(self, goal_handle, center, hold_s, radius):
        '''Wait until the drone has stayed inside the sphere for hold_s seconds in a row.
        Leaving the sphere or losing the marker restarts the count. Returns 'ok' | 'cancel' | 'shutdown'.'''
        hold_start = None
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                return 'cancel'
            self.send_feedback(goal_handle)
            now = time.monotonic()
            if self.marker_fresh() and math.dist(self.current_state, center) <= radius:
                if hold_start is None:
                    hold_start = now
                if now - hold_start >= hold_s:
                    return 'ok'
            else:
                hold_start = None
            time.sleep(0.02)
        return 'shutdown'

    def land(self, goal_handle):
        '''Slow, controlled descent to the ground, then disarm. Returns 'ok' | 'cancel' | 'shutdown'.'''
        ground = self.ground_z if self.ground_z is not None else self.current_state[2] + 2.0
        z_start = self.current_state[2]
        t0 = time.monotonic()
        max_time = (ground - z_start) / self.descent_rate + 8.0
        self.get_logger().info('Landing: descending from Z=%.2f to the ground at Z=%.2f' % (z_start, ground))
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                return 'cancel'
            elapsed = time.monotonic() - t0
            self.desired_state[2] = min(z_start + self.descent_rate * elapsed, ground)
            self.send_feedback(goal_handle)
            touched = self.current_state[2] >= ground - self.touchdown_margin and self.desired_state[2] >= ground
            if touched or elapsed > max_time:
                self.landed = True  # pid() now keeps the motors disarmed
                self.disarm()
                self.get_logger().info('Landed (Z=%.2f)' % self.current_state[2])
                return 'ok'
            time.sleep(0.02)
        return 'shutdown'

    def finish_cancelled(self, goal_handle):
        # Hold exactly where the drone is until the next goal arrives
        self.desired_state = list(self.current_state)
        goal_handle.canceled()
        self.get_logger().info('Goal cancelled - holding at (%.2f, %.2f, %.2f)' % tuple(self.current_state))
        return NavToWaypoint.Result()

    # ------------------------------------------------------------------ action
    def execute_callback(self, goal_handle):
        req = goal_handle.request
        wp = req.waypoint.position
        if not req.hover:
            mode = 'pass'
        elif req.waypoint.orientation.w > 1.5:
            mode = 'home'
        else:
            mode = 'survivor'

        self.landed = False
        self.desired_state = [wp.x, wp.y, wp.z]
        self.have_goal = True
        self.get_logger().info('Goal (%s): (%.2f, %.2f, %.2f)' % (mode, wp.x, wp.y, wp.z))
        start = time.monotonic()

        def finish(outcome):
            if outcome == 'cancel':
                return self.finish_cancelled(goal_handle)
            if outcome != 'ok':
                goal_handle.abort()
                return NavToWaypoint.Result()
            goal_handle.succeed()
            result = NavToWaypoint.Result()
            result.hov_time = int(round(time.monotonic() - start))
            self.get_logger().info('Goal done (%d s)' % result.hov_time)
            return result

        if mode == 'pass':
            return finish(self.hold_in_sphere(goal_handle, [wp.x, wp.y, wp.z], 0.0, self.pass_radius))

        # Visit: hold at Z = 13 for 1 s
        outcome = self.hold_in_sphere(goal_handle, [wp.x, wp.y, wp.z], self.visit_hold if mode == 'survivor'
                                      else self.home_hold, self.sphere_radius)
        if outcome != 'ok':
            return finish(outcome)

        if mode == 'survivor':
            # Save: descend and hold at Z = 14 for 2 s
            self.desired_state[2] = self.save_z
            self.get_logger().info('Visit done - saving at Z=%.1f' % self.save_z)
            return finish(self.hold_in_sphere(goal_handle, [wp.x, wp.y, self.save_z], self.save_hold,
                                              self.sphere_radius))

        # home: hovered at Z = 13 for 1 s, now land
        return finish(self.land(goal_handle))


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
