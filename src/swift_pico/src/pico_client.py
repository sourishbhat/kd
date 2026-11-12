#!/usr/bin/env python3
'''
Task 3A action client.

1. Gets the critical route from the service 'waypoints' (GetWaypoints).
2. Waits for the action server, then starts the critical-window clock (take-off is the first goal)
   and sends the critical goals one at a time.
3. THE SWITCH: when the last critical survivor is saved, or the window (default 55 s) runs out,
   whichever is first: cancel the goal in flight, send the drone's last reported position to the service
   'plan_stable' (PlanStable) and fly the stable route it returns. Its last goal is F6, where the server
   hovers and lands.

Parameter (for the "force the Switch early" checkpoint):
    ros2 run swift_pico pico_client --ros-args -p critical_window:=6.0     # set back to 55.0 for the real run
'''

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Point

from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints, PlanStable


class MissionClient(Node):
    def __init__(self):
        super().__init__('waypoint_client')
        self.declare_parameter('critical_window', 55.0)
        self.critical_window = float(self.get_parameter('critical_window').value)

        self.critical_client = self.create_client(GetWaypoints, 'waypoints')
        self.stable_client = self.create_client(PlanStable, 'plan_stable')
        self.action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')

        self.last_position = None   # (x, y, z) from the latest action feedback
        self._last_log = 0.0

    # ------------------------------------------------------------------ services
    def call_service(self, client, request, name, attempts=20):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the %s service...' % name)
        for attempt in range(attempts):
            future = client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response is not None and len(response.waypoints.poses) > 0:
                return response
            self.get_logger().info('No %s route yet (attempt %d/%d), retrying...' % (name, attempt + 1, attempts))
            time.sleep(1.0)
        return None

    # ------------------------------------------------------------------ flying
    def feedback_callback(self, feedback_msg):
        p = feedback_msg.feedback.current_waypoint.pose.position
        self.last_position = (p.x, p.y, p.z)
        now = time.monotonic()
        if now - self._last_log >= 1.0:
            self._last_log = now
            self.get_logger().info('Feedback: (%.2f, %.2f, %.2f)' % self.last_position)

    def make_goal(self, pose, marker_id):
        goal = NavToWaypoint.Goal()
        goal.waypoint = pose
        goal.hover = not marker_id.startswith('via')            # survivors and home hold; via points don't
        goal.waypoint.orientation.w = 2.0 if marker_id.startswith('home') else 1.0   # 2.0 = hover then land
        return goal

    def cancel(self, goal_handle, result_future):
        cancel_future = goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=5.0)

    def fly(self, pose, marker_id, deadline=None):
        '''Send one goal and wait for it. Returns 'done', 'timeout' (cancelled at the deadline) or 'failed'.'''
        if deadline is not None and time.monotonic() >= deadline:
            return 'timeout'
        goal = self.make_goal(pose, marker_id)
        self.get_logger().info('Sending %s: (%.2f, %.2f, %.2f)' % (
            marker_id, pose.position.x, pose.position.y, pose.position.z))

        send_future = self.action_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('%s was rejected by the server' % marker_id)
            return 'failed'

        result_future = goal_handle.get_result_async()
        while rclpy.ok():
            wait = 0.1
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self.get_logger().warn('Critical window over - cancelling %s' % marker_id)
                    self.cancel(goal_handle, result_future)
                    return 'timeout'
                wait = min(wait, remaining)
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=wait)
            if result_future.done():
                return 'done'
        return 'failed'

    # ------------------------------------------------------------------ mission
    def run(self):
        crit = self.call_service(self.critical_client, self._critical_request(), 'critical')
        if crit is None:
            self.get_logger().error('Could not get the critical route')
            return
        ids, poses = list(crit.marker_ids), list(crit.waypoints.poses)
        self.get_logger().info('Critical route: %s' % ' -> '.join(ids))

        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for the action server...')

        # Take-off is the first goal: start the clock only now, not when this node launched
        t0 = time.monotonic()
        deadline = t0 + self.critical_window
        self.get_logger().info('Critical window started: %.0f s' % self.critical_window)

        switch_reason = 'all critical survivors saved'
        for pose, mid in zip(poses, ids):
            outcome = self.fly(pose, mid, deadline)
            if outcome == 'timeout':
                switch_reason = 'critical window over'
                break
            if outcome == 'failed':
                self.get_logger().error('Stopping: %s failed' % mid)
                return

        # ---------------- THE SWITCH ----------------
        self.get_logger().info('SWITCH at t = %.1f s (%s)' % (time.monotonic() - t0, switch_reason))
        position = self.last_position if self.last_position is not None else (0.0, 0.0, 13.0)
        request = PlanStable.Request()
        request.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
        stable = self.call_service(self.stable_client, request, 'stable')
        if stable is None:
            self.get_logger().error('Could not get the stable route')
            return
        ids, poses = list(stable.marker_ids), list(stable.waypoints.poses)
        self.get_logger().info('Stable route: %s' % ' -> '.join(ids))

        for pose, mid in zip(poses, ids):
            if self.fly(pose, mid) != 'done':
                self.get_logger().error('Stopping: %s failed' % mid)
                return

        self.get_logger().info('Mission complete at t = %.1f s: all survivors visited, landed on F6.'
                               % (time.monotonic() - t0))

    @staticmethod
    def _critical_request():
        request = GetWaypoints.Request()
        request.get_waypoints = True
        return request


def main(args=None):
    rclpy.init(args=args)
    client = MissionClient()
    try:
        client.run()
    except KeyboardInterrupt:
        client.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
