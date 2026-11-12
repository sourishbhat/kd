#!/usr/bin/env python3
'''
Waypoint service: returns the four predefined waypoints, always in the order wp1 -> wp4.
Service name: 'waypoints'  (type: waypoint_navigation/srv/GetWaypoints)
'''

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from waypoint_navigation.srv import GetWaypoints


class WayPoints(Node):
    def __init__(self):
        super().__init__('waypoints_service')

        self.waypoint_service = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback)

        # ORDER MATTERS: wp1 first, wp4 last. [x, y, z] in whycode units
        self.waypoints = [
            [-2.41, -2.41, 13.0],  # wp1
            [2.41, 1.20, 13.0],    # wp2
            [1.20, -2.41, 13.0],   # wp3
            [-1.20, 1.20, 13.0],   # wp4
        ]
        self.marker_ids = ['wp1', 'wp2', 'wp3', 'wp4']

    def waypoint_callback(self, request, response):
        if not request.get_waypoints:
            self.get_logger().info('Request rejected (get_waypoints is False)')
            return response  # empty response

        poses = []
        for x, y, z in self.waypoints:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.w = 1.0
            poses.append(pose)

        response.waypoints.header.frame_id = 'whycode'
        response.waypoints.header.stamp = self.get_clock().now().to_msg()
        response.waypoints.poses = poses
        response.marker_ids = list(self.marker_ids)
        self.get_logger().info('Incoming request for Waypoints: sent %d waypoints' % len(poses))
        return response


def main():
    rclpy.init()
    node = WayPoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
