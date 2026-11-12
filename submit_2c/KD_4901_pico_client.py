#!/usr/bin/env python3
'''
Action client: asks the waypoint service for the waypoints, then sends them one at a time,
in the order returned, to the action server 'waypoint_navigation'. The next goal is only sent
after the server reports the previous waypoint as reached and stabilised.
'''

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from waypoint_navigation.action import NavToWaypoint
from waypoint_navigation.srv import GetWaypoints


class WayPointClient(Node):
    def __init__(self):
        super().__init__('waypoint_client')
        self.srv_client = self.create_client(GetWaypoints, 'waypoints')
        self.action_client = ActionClient(self, NavToWaypoint, 'waypoint_navigation')
        self._last_fb_log = 0.0

    def get_waypoints(self, attempts=20):
        while not self.srv_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for waypoint service...')
        # The route may not be planned yet (service returns an empty list) -> ask again
        for attempt in range(attempts):
            request = GetWaypoints.Request()
            request.get_waypoints = True
            future = self.srv_client.call_async(request)
            rclpy.spin_until_future_complete(self, future)
            response = future.result()
            if response is not None and len(response.waypoints.poses) > 0:
                return response
            self.get_logger().info('No route yet (attempt %d/%d), retrying...' % (attempt + 1, attempts))
            time.sleep(1.0)
        return None

    def feedback_callback(self, feedback_msg):
        # Log at most once per second to keep the terminal readable
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_fb_log < 1.0:
            return
        self._last_fb_log = now
        p = feedback_msg.feedback.current_waypoint.pose.position
        self.get_logger().info('Feedback: current position (%.2f, %.2f, %.2f)' % (p.x, p.y, p.z))

    def go_to(self, pose, name, hover=True):
        # The server may not be up yet: wait for it instead of failing
        while not self.action_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for action server...')

        goal = NavToWaypoint.Goal()
        goal.waypoint = pose
        goal.hover = hover

        self.get_logger().info('Sending %s: (%.2f, %.2f, %.2f)' % (
            name, pose.position.x, pose.position.y, pose.position.z))
        send_future = self.action_client.send_goal_async(goal, feedback_callback=self.feedback_callback)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('%s was rejected by the server' % name)
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        self.get_logger().info('%s reached and stabilised (took %d s)' % (name, result.hov_time))
        return True


def main(args=None):
    rclpy.init(args=args)
    client = WayPointClient()
    try:
        response = client.get_waypoints()
        if response is None or len(response.waypoints.poses) == 0:
            client.get_logger().error('Service returned no waypoints')
            return

        names = list(response.marker_ids)
        for i, pose in enumerate(response.waypoints.poses):  # same order as returned, none skipped
            name = names[i] if i < len(names) else 'wp%d' % (i + 1)
            hover = not name.startswith('via')  # only survivors ('critical:...') get the 3 s hover
            if not client.go_to(pose, name, hover):
                client.get_logger().error('Stopping: %s failed' % name)
                return
        client.get_logger().info('All %d waypoints reached. Navigation complete.' % len(response.waypoints.poses))
    except KeyboardInterrupt:
        client.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
