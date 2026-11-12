#!/usr/bin/env python3

'''
This python file runs a ROS 2-node of name pico_control which holds the position of Swift Pico Drone on the given drone.
This node publishes and subsribes the following topics:

        PUBLICATIONS            SUBSCRIPTIONS
        /drone_command          /whycode_node/markers
        /pos_error              /throttle_pid
                                /pitch_pid
                                /roll_pid

Axis convention used throughout this file (index -> axis -> command):
        0 -> X -> pitch
        1 -> Y -> roll
        2 -> Z -> throttle
'''

# Importing the required libraries

from swift_msgs.msg import SwiftMsgs
from whycode_interfaces.msg import MarkerArray
from controller_msg.msg import PIDTune
from error_msg.msg import Error
from std_msgs.msg import Header
import time
import rclpy
from rclpy.node import Node


class Swift_Pico(Node):
    def __init__(self):
        super().__init__('pico_controller')  # initializing ros node with name pico_controller

        # Current position of the drone [x, y, z] (raw WhyCode units). Updated in whycode_callback
        self.current_state = [0.0, 0.0, 0.0]
        self.marker_received = False

        # Setpoint to reach and hold [x, y, z]
        self.desired_state = [1.5, -1.5, 13.0]

        # Declaring a cmd of message type swift_msgs and initializing values
        self.cmd = SwiftMsgs()
        self.cmd.rc_roll = 1500
        self.cmd.rc_pitch = 1500
        self.cmd.rc_yaw = 1500
        self.cmd.rc_throttle = 1500

        # PID gains, indexed [x/pitch, y/roll, z/throttle].
        # These are overwritten live by the tuner GUI. Once tuned, hardcode your final values here
        # (and comment out nothing else: the subscribers are harmless without the GUI).
        # Starting values = scaled results from the GUI: [x/pitch, y/roll, z/throttle]
        # Altitude verified (TEST STEP 1). TEST STEP 2: pitch/roll now enabled with moderate gains.
        # If an axis' error GROWS instead of shrinking, flip that axis' entry in self.sign.
        self.Kp = [30.0, 30.0, 100.0]
        self.Ki = [0.0, 0.0, 12.0]
        self.Kd = [60.0, 60.0, 80.0]

        # GUI slider -> real gain scaling, per axis [x/pitch, y/roll, z/throttle]
        self.kp_scale = [0.03, 0.03, 0.03]
        self.ki_scale = [0.008, 0.008, 0.008]
        self.kd_scale = [0.6, 0.6, 0.6]

        # ----------------------- Other required variables for PID -------------------------------------------
        self.error = [0.0, 0.0, 0.0]
        self.prev_error = [0.0, 0.0, 0.0]
        self.error_sum = [0.0, 0.0, 0.0]
        self.out = [0.0, 0.0, 0.0]

        # Sign of each axis' output on the command. error = setpoint - current.
        # Throttle: the camera is overhead, so larger Z = LOWER drone. Reaching a smaller Z setpoint
        # means climbing, i.e. a negative error needs MORE throttle -> sign -1.
        # Pitch/roll signs depend on how WhyCode axes map to the drone's body: EXPERIMENT and flip
        # to -1 for an axis if the drone runs away from the setpoint along that axis.
        self.sign = [1, 1, -1]

        # Limits
        # Pitch/roll limited to a modest tilt so the drone can't flip or lose lift; throttle uses full range
        self.max_values = [1620, 1620, 2000]
        self.min_values = [1380, 1380, 1000]
        self.neutral = 1500

        # Anti-windup: cap on the magnitude of the integral CONTRIBUTION (Ki * sum) per axis
        self.max_integral_out = [60.0, 60.0, 60.0]

        self.last_time = None

        # Low-pass filtered derivative (0 < alpha <= 1; smaller = smoother) to tame noise/kicks
        self.d_filt = [0.0, 0.0, 0.0]
        self.d_alpha = 0.2

        # If the marker hasn't been seen for this long, stop trusting stale data
        self.last_marker_time = None
        self.marker_timeout = 0.3  # seconds
        # ----------------------------------------------------------------------------------------------------

        # Sample time in which PID runs
        self.sample_time = 0.033  # in seconds

        # Publishing /drone_command, /pos_error
        self.command_pub = self.create_publisher(SwiftMsgs, '/drone_command', 10)
        self.pos_error_pub = self.create_publisher(Error, '/pos_error', 10)
        self.heartbeat_pub = self.create_publisher(Header, '/controller_heartbeat', 10)

        # Subscribing to /whycode_node/markers, /throttle_pid, /pitch_pid, /roll_pid
        self.create_subscription(MarkerArray, '/whycode_node/markers', self.whycode_callback, 1)
        self.create_subscription(PIDTune, "/throttle_pid", self.altitude_set_pid, 1)
        self.create_subscription(PIDTune, "/pitch_pid", self.pitch_set_pid, 1)
        self.create_subscription(PIDTune, "/roll_pid", self.roll_set_pid, 1)

        self.publish_heartbeat()
        self.create_timer(0.1, self.publish_heartbeat)

        time.sleep(0.5)  # let the heartbeat reach the rosbag before arming
        self.arm()  # ARMING THE DRONE

        # Run the PID loop at a fixed interval
        self.create_timer(self.sample_time, self.pid)

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
        self.command_pub.publish(self.cmd)  # Publishing /drone_command

    # Whycode callback: runs each time /whycode_node publishes /whycode_node/markers
    def whycode_callback(self, msg):
        if not msg.markers:  # marker lost from the camera: keep last known state
            return
        pos = msg.markers[0].position.position
        self.current_state[0] = pos.x
        self.current_state[1] = pos.y
        self.current_state[2] = pos.z
        self.marker_received = True
        self.last_marker_time = time.monotonic()

    # Tuner GUI callbacks. Index: 0 = x/pitch, 1 = y/roll, 2 = z/throttle
    def set_pid(self, axis, msg):
        self.Kp[axis] = msg.kp * self.kp_scale[axis]
        self.Ki[axis] = msg.ki * self.ki_scale[axis]
        self.Kd[axis] = msg.kd * self.kd_scale[axis]

    def altitude_set_pid(self, alt):
        self.set_pid(2, alt)

    def pitch_set_pid(self, pitch):
        self.set_pid(0, pitch)

    def roll_set_pid(self, roll):
        self.set_pid(1, roll)

    @staticmethod
    def clamp(value, low, high):
        return max(low, min(high, value))

    def pid(self):
        # No marker -> no feedback -> don't command anything based on stale/zero data
        if not self.marker_received:
            return

        # Marker lost: hold neutral (hover) command and reset state instead of flying on stale data
        if time.monotonic() - self.last_marker_time > self.marker_timeout:
            self.cmd.rc_roll = self.cmd.rc_pitch = self.cmd.rc_throttle = self.neutral
            self.command_pub.publish(self.cmd)
            self.error_sum = [0.0, 0.0, 0.0]
            self.d_filt = [0.0, 0.0, 0.0]
            self.last_time = None
            self.get_logger().warn('Marker lost - holding neutral', throttle_duration_sec=1.0)
            return

        # Actual elapsed time (timer jitter makes this slightly different from sample_time)
        now = time.monotonic()
        if self.last_time is None:
            self.last_time = now
            self.prev_error = [self.desired_state[i] - self.current_state[i] for i in range(3)]
            return
        dt = now - self.last_time
        self.last_time = now
        if dt <= 0.0:
            return

        for i in range(3):
            # 1. Error: setpoint - current
            self.error[i] = self.desired_state[i] - self.current_state[i]

            # 2. Integral with anti-windup (clamp the integral so Ki*sum stays bounded)
            self.error_sum[i] += self.error[i] * dt
            if self.Ki[i] != 0.0:
                limit = self.max_integral_out[i] / self.Ki[i]
                self.error_sum[i] = self.clamp(self.error_sum[i], -limit, limit)

            # 3. Derivative
            d_raw = (self.error[i] - self.prev_error[i]) / dt
            self.d_filt[i] = self.d_alpha * d_raw + (1.0 - self.d_alpha) * self.d_filt[i]
            d_error = self.d_filt[i]

            # 4. PID output
            self.out[i] = (self.Kp[i] * self.error[i]
                           + self.Ki[i] * self.error_sum[i]
                           + self.Kd[i] * d_error)

            # 7. Update previous error
            self.prev_error[i] = self.error[i]

        # 5. Map outputs onto the commands around neutral (index 0 pitch, 1 roll, 2 throttle)
        pitch_cmd = self.neutral + self.sign[0] * self.out[0]
        roll_cmd = self.neutral + self.sign[1] * self.out[1]
        throttle_cmd = self.neutral + self.sign[2] * self.out[2]

        # 6. Limit to valid range before publishing
        self.cmd.rc_pitch = int(self.clamp(pitch_cmd, self.min_values[0], self.max_values[0]))
        self.cmd.rc_roll = int(self.clamp(roll_cmd, self.min_values[1], self.max_values[1]))
        self.cmd.rc_throttle = int(self.clamp(throttle_cmd, self.min_values[2], self.max_values[2]))

        self.command_pub.publish(self.cmd)

        # Publish errors for the tuner GUI and rosbag
        pos_error = Error()
        pos_error.pitch_error = float(self.error[0])
        pos_error.roll_error = float(self.error[1])
        pos_error.throttle_error = float(self.error[2])
        self.pos_error_pub.publish(pos_error)


def main(args=None):
    rclpy.init(args=args)
    swift_pico = Swift_Pico()

    try:
        rclpy.spin(swift_pico)
    except KeyboardInterrupt:
        swift_pico.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        swift_pico.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
