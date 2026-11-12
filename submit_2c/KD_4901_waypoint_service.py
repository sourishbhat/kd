#!/usr/bin/env python3
'''
Task 2C waypoint service.

1. Subscribes ONCE to /image_raw (latched frame: RELIABLE + TRANSIENT_LOCAL), then drops the subscription.
2. Runs the Task 1A pipeline on that frame (ArUco corners -> rectify -> colour/shape detection),
   keeping only the CRITICAL survivors (red triangles), each named by its nearest grid intersection.
3. Looks every intersection up in grid_whycode_coordinates.csv (Z is always 13).
4. Plans the shortest visiting order (all permutations) starting from the arena centre F6.
5. Serves the finished route on the service 'waypoints' (waypoint_navigation/srv/GetWaypoints).

marker_ids tags every waypoint so the client knows whether to hover:
    'critical:<intersection>'  -> survivor, hover 3 s (compulsory)
    'via:<intersection>'       -> intermediate point, fly straight past

Parameters (ros2 run swift_pico waypoint_service --ros-args -p route_mode:=grid):
    route_mode : 'direct' (default, straight legs, fastest) or 'grid' (legs follow grid lines, one corner stop)
    csv_path   : override path of the coordinates file (.json or .csv)
    debug_dir  : where the annotated frame is saved (default ~/pico_ws/task2c_debug)
'''

import csv
import itertools
import json
import math
import os
import re
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from waypoint_navigation.srv import GetWaypoints

# ----------------------------------------------------------------------------- Task 1A config
REQUIRED_MARKER_IDS = {80, 85, 90, 95}
RECT_SIZE = 900
GRID_CELLS = 12
MIN_SURVIVOR_AREA = 120
MAX_SURVIVOR_AREA = 100000
CIRCULARITY_MIN = 0.65
TRIANGLE_EPSILON = 0.04
DUPLICATE_DIST = 40

YELLOW_LOWER = np.array([18, 90, 90])
YELLOW_UPPER = np.array([35, 255, 255])
RED_LOWER_1 = np.array([0, 110, 70])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 110, 70])
RED_UPPER_2 = np.array([180, 255, 255])

START_INTERSECTION = 'F6'   # the drone starts at the arena centre
FLIGHT_Z = 13.0

NAME_RE = re.compile(r'^([A-Z]+)(\d+)$')


# ============================================================================= coordinate lookup (JSON or CSV)
def _num_pair(value):
    '''(x, y) from [x, y, ...] or {'x': .., 'y': ..}; None if value isn't a coordinate.'''
    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value[:2]):
        return float(value[0]), float(value[1])
    if isinstance(value, dict):
        low = {str(k).lower(): v for k, v in value.items()}
        if all(isinstance(low.get(k), (int, float)) for k in ('x', 'y')):
            return float(low['x']), float(low['y'])
    return None


def load_grid_json(path):
    '''Returns {'A1': (x, y), ...} from a JSON file. Handles the common shapes:
        {"D5": [x, y]}  {"D5": {"x": .., "y": ..}}  [{"name": "D5", "x": .., "y": ..}, ...]
        {"D": {"5": [x, y]}}  and  {"5": {"D": [x, y]}}  (column/row nested either way)'''
    with open(path) as f:
        data = json.load(f)

    coords = {}

    def add(name_parts, pair):
        # Use only the last one or two keys, so wrapper keys such as "grid" are ignored
        for n in (1, 2):
            name = ''.join(name_parts[-n:]).upper().replace(' ', '')
            if re.match(r'^[A-Z]\d+$', name):
                coords[name] = pair
                return
            m = re.match(r'^(\d+)([A-Z])$', name)  # row-first nesting, e.g. "5" then "D"
            if m:
                coords['%s%s' % (m.group(2), m.group(1))] = pair
                return

    def walk(node, path):
        pair = _num_pair(node)
        if pair is not None and path:
            add(path, pair)
            return
        if isinstance(node, dict):
            low = {str(k).lower(): v for k, v in node.items()}
            label = next((low[k] for k in ('name', 'intersection', 'label', 'id', 'grid', 'point')
                          if isinstance(low.get(k), str)), None)
            if label:
                pair = _num_pair(node)
                if pair is None:  # coordinates nested under another key, e.g. "coordinates": [x, y]
                    pair = next((q for q in (_num_pair(v) for v in node.values()) if q is not None), None)
                if pair is not None:
                    add([label], pair)
                    return
            for k, v in node.items():
                walk(v, path + [str(k)])
        elif isinstance(node, list):
            for item in node:
                walk(item, path)

    walk(data, [])
    if not coords:
        raise ValueError('no intersections like "D5" with x/y found in %s' % path)
    return coords


# ============================================================================= CSV lookup
def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def load_grid_csv(path):
    '''Returns {'A1': (x, y), ...}. Tolerates a header row and different column orders.'''
    with open(path, newline='') as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError('coordinates CSV is empty')

    name_i, x_i, y_i, start = 0, 1, 2, 0
    first = rows[0]
    if not any(_is_number(c) for c in first):  # header row present
        start = 1
        low = [c.lower() for c in first]

        def find(pred, default):
            for i, h in enumerate(low):
                if pred(h):
                    return i
            return default

        x_i = find(lambda h: h in ('x', 'whycode_x', 'x_whycode') or h.endswith('_x') or h.startswith('x_'), 1)
        y_i = find(lambda h: h in ('y', 'whycode_y', 'y_whycode') or h.endswith('_y') or h.startswith('y_'), 2)
        name_i = find(lambda h: any(k in h for k in ('inter', 'name', 'label', 'grid', 'id', 'point')), 0)

    coords = {}
    for r in rows[start:]:
        try:
            name = r[name_i].upper().replace(' ', '')
            if NAME_RE.match(name):
                coords[name] = (float(r[x_i]), float(r[y_i]))
        except (IndexError, ValueError):
            continue
    if not coords:
        raise ValueError('no intersection rows (like "D5,x,y") found in CSV; first row was %r' % (first,))
    return coords


# ============================================================================= Task 1A pipeline (pure functions)
def image_msg_to_bgr(msg):
    enc = msg.encoding.lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc)
    if channels is None:
        channels = max(1, msg.step // msg.width)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.step)
    arr = buf[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if enc == 'rgb8':
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if enc == 'rgba8':
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if enc == 'bgra8':
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(arr)


def detect_aruco_markers(gray):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None:
        return [], None
    return corners, ids.flatten()


def get_ordered_arena_corners(marker_corners):
    centers = {mid: pts.mean(axis=0) for mid, pts in marker_corners.items()}
    arena_center = np.mean(list(centers.values()), axis=0)

    inner = {}
    for mid, pts in marker_corners.items():
        inner[mid] = pts[np.argmin(np.linalg.norm(pts - arena_center, axis=1))]

    quadrant = {}
    for mid, c in centers.items():
        dx, dy = c - arena_center
        if dx < 0 and dy < 0:
            quadrant['TL'] = mid
        elif dx > 0 and dy < 0:
            quadrant['TR'] = mid
        elif dx > 0 and dy > 0:
            quadrant['BR'] = mid
        else:
            quadrant['BL'] = mid
    if set(quadrant) != {'TL', 'TR', 'BR', 'BL'}:
        raise ValueError('could not determine the four arena corners from the markers')
    return np.array([inner[quadrant[k]] for k in ('TL', 'TR', 'BR', 'BL')], dtype=np.float32)


def rectify_arena(image, src_corners, size=RECT_SIZE):
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners, dst)
    return cv2.warpPerspective(image, M, (size, size))


def build_grid_intersections(size=RECT_SIZE, cells=GRID_CELLS):
    '''[(name, px, py)] for the 11x11 interior intersections: columns A-K, rows 1-11.'''
    step = size / cells
    pts = []
    for row in range(cells - 1):
        for col in range(cells - 1):
            pts.append(('%s%d' % (chr(ord('A') + col), row + 1), (col + 1) * step, (row + 1) * step))
    return pts


def is_triangle(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return False
    return len(cv2.approxPolyDP(contour, TRIANGLE_EPSILON * perimeter, True)) == 3


def is_circle(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if area <= 0 or perimeter <= 0:
        return False
    return 4.0 * np.pi * area / (perimeter * perimeter) > CIRCULARITY_MIN


def contour_center(contour):
    M = cv2.moments(contour)
    if M['m00'] != 0:
        return M['m10'] / M['m00'], M['m01'] / M['m00']
    cx, cy = contour.reshape(-1, 2).mean(axis=0)
    return float(cx), float(cy)


def remove_duplicates(points, min_dist=DUPLICATE_DIST):
    out = []
    for p in points:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= min_dist for q in out):
            out.append(p)
    return out


def find_color_centers(warped, lowers, uppers, shape_filter=None):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in zip(lowers, uppers):
        mask |= cv2.inRange(hsv, lo, hi)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers, kept = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SURVIVOR_AREA or area > MAX_SURVIVOR_AREA:
            continue
        if shape_filter is not None and not shape_filter(cnt):
            continue
        centers.append(contour_center(cnt))
        kept.append(cnt)
    return remove_duplicates(centers), kept


def nearest_intersection_names(centers, grid_points):
    names = [p[0] for p in grid_points]
    coords = np.array([[p[1], p[2]] for p in grid_points], dtype=np.float64)
    labels = []
    for cx, cy in centers:
        d2 = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
        labels.append(names[int(np.argmin(d2))])
    return labels


def detect_critical(image):
    '''Returns (critical_labels, critical_centers, critical_contours, warped, used_fallback).'''
    used_fallback = False
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_aruco_markers(gray)
        found = set(int(i) for i in ids) if ids is not None else set()
        missing = REQUIRED_MARKER_IDS - found
        if missing:
            raise ValueError('missing corner marker ID(s) %s' % sorted(missing))
        marker_corners = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids) if int(i) in REQUIRED_MARKER_IDS}
        warped = rectify_arena(image, get_ordered_arena_corners(marker_corners))
    except ValueError:
        # Corner markers not usable: assume the frame already is the arena
        used_fallback = True
        warped = cv2.resize(image, (RECT_SIZE, RECT_SIZE))

    grid_points = build_grid_intersections()
    centers, contours = find_color_centers(
        warped, [RED_LOWER_1, RED_LOWER_2], [RED_UPPER_1, RED_UPPER_2], shape_filter=is_triangle)
    labels = nearest_intersection_names(centers, grid_points)
    return labels, centers, contours, warped, used_fallback


# ============================================================================= route planning (pure functions)
def split_name(name):
    m = NAME_RE.match(name)
    return m.group(1), int(m.group(2))


def leg_cost(a, b, mode):
    '''a, b are (x, y). direct = straight line; grid = along grid lines (Manhattan).'''
    dx, dy = b[0] - a[0], b[1] - a[1]
    return abs(dx) + abs(dy) if mode == 'grid' else math.hypot(dx, dy)


def best_order(start_xy, targets, coords, mode):
    '''targets: list of intersection names. Tries every visiting order (24 for four survivors).'''
    if len(targets) > 8:  # keep it fast if detection ever returns lots of points
        order, cur, left = [], start_xy, list(targets)
        while left:
            nxt = min(left, key=lambda t: leg_cost(cur, coords[t], mode))
            order.append(nxt)
            cur = coords[nxt]
            left.remove(nxt)
        return order

    best, best_cost = None, float('inf')
    for perm in itertools.permutations(targets):
        cost, cur = 0.0, start_xy
        for t in perm:
            cost += leg_cost(cur, coords[t], mode)
            cur = coords[t]
        if cost < best_cost:
            best, best_cost = list(perm), cost
    return best


def build_route(start_name, targets, coords, mode):
    '''Returns [(tag, intersection_name, x, y)] in visiting order. tag is 'critical' or 'via'.'''
    order = best_order(coords[start_name], targets, coords, mode)
    route, cur_name = [], start_name
    for t in order:
        if mode == 'grid':
            col_a, row_a = split_name(cur_name)
            col_b, row_b = split_name(t)
            corner = '%s%d' % (col_b, row_a)  # fly along the row first, then down/up the column
            if corner not in (cur_name, t) and corner in coords:
                route.append(('via', corner, coords[corner][0], coords[corner][1]))
        route.append(('critical', t, coords[t][0], coords[t][1]))
        cur_name = t
    return route


# ============================================================================= ROS node
class WayPoints(Node):
    def __init__(self):
        super().__init__('waypoints_service')

        self.declare_parameter('route_mode', 'direct')
        self.declare_parameter('csv_path', '')
        self.declare_parameter('debug_dir', os.path.expanduser('~/pico_ws/task2c_debug'))
        self.declare_parameter('request_wait_s', 25.0)

        self.route_mode = self.get_parameter('route_mode').value
        self.debug_dir = self.get_parameter('debug_dir').value
        self.request_wait_s = float(self.get_parameter('request_wait_s').value)

        self.grid = self.load_coordinates()

        self.waypoints = []       # [(x, y, z)] filled by the image callback, in visiting order
        self.marker_ids = []      # 'critical:D5' / 'via:H5'
        self.ready = threading.Event()

        group = ReentrantCallbackGroup()

        # The frame is published once and latched -> RELIABLE + TRANSIENT_LOCAL so we still get it
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos,
                                                  callback_group=group)

        self.waypoint_service = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback,
                                                    callback_group=group)
        self.get_logger().info('Waypoint service up (route_mode=%s). Waiting for /image_raw frame...' % self.route_mode)

    # ------------------------------------------------------------------ setup helpers
    def load_coordinates(self):
        names = ('grid_whycode_coordinates.json', 'grid_whycode_coordinates.csv')
        candidates = []
        param_path = self.get_parameter('csv_path').value
        if param_path:
            candidates.append(os.path.expanduser(param_path))
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('swift_pico')
            for n in names:
                candidates += [os.path.join(share, 'scripts', n), os.path.join(share, n)]
        except Exception:
            pass
        base = os.path.expanduser('~/pico_ws/src/swift_pico')
        for n in names:
            candidates += [os.path.join(base, 'scripts', n), os.path.join(base, 'src', n)]

        for path in candidates:
            if os.path.isfile(path):
                coords = load_grid_json(path) if path.lower().endswith('.json') else load_grid_csv(path)
                self.get_logger().info('Loaded %d intersections from %s' % (len(coords), path))
                return coords
        raise FileNotFoundError('grid_whycode_coordinates (.json/.csv) not found. Tried: %s' % candidates)

    # ------------------------------------------------------------------ image -> route
    def image_callback(self, msg):
        try:
            image = image_msg_to_bgr(msg)
            labels, centers, contours, warped, used_fallback = detect_critical(image)
            if used_fallback:
                self.get_logger().warn('Corner markers not found - assumed the whole frame is the arena')
            self.get_logger().info('Critical survivors detected: %s' % sorted(labels))
            if len(labels) != 4:
                self.get_logger().warn('Expected 4 critical survivors, found %d - check the debug image' % len(labels))

            targets = []
            for name in labels:
                if name in self.grid and name not in targets:
                    targets.append(name)
                else:
                    self.get_logger().error('Intersection %s missing from the CSV - skipping it' % name)
            if not targets:
                raise RuntimeError('no usable critical survivors detected')

            route = build_route(START_INTERSECTION, targets, self.grid, self.route_mode)

            self.waypoints = [(x, y, FLIGHT_Z) for (_, _, x, y) in route]
            self.marker_ids = ['%s:%s' % (tag, name) for (tag, name, _, _) in route]
            self.get_logger().info('Planned route from %s: %s' % (
                START_INTERSECTION, ' -> '.join(self.marker_ids)))
            for mid, (x, y, z) in zip(self.marker_ids, self.waypoints):
                self.get_logger().info('  %-14s (%.2f, %.2f, %.2f)' % (mid, x, y, z))

            self.save_debug(warped, contours, centers, labels, route)
            self.ready.set()
        except Exception as exc:  # keep the node alive; requests will report an empty list
            self.get_logger().error('Route planning failed: %s' % exc)
        finally:
            # One frame is all we need
            self.destroy_subscription(self.image_sub)

    def save_debug(self, warped, contours, centers, labels, route):
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            img = warped.copy()
            step = RECT_SIZE / GRID_CELLS
            for i in range(GRID_CELLS + 1):
                p = int(round(i * step))
                cv2.line(img, (p, 0), (p, RECT_SIZE - 1), (0, 255, 0), 1)
                cv2.line(img, (0, p), (RECT_SIZE - 1, p), (0, 255, 0), 1)
            cv2.drawContours(img, contours, -1, (255, 0, 255), 2)

            px = {n: (int(x), int(y)) for n, x, y in build_grid_intersections()}
            path = [px[START_INTERSECTION]] + [px[n] for (_, n, _, _) in route]
            cv2.polylines(img, [np.array(path, dtype=np.int32)], False, (0, 140, 255), 3)
            cv2.circle(img, path[0], 8, (255, 0, 0), -1)
            n = 0
            for (tag, name, _, _) in route:
                pt = px[name]
                if tag == 'critical':
                    n += 1
                    cv2.circle(img, pt, 12, (0, 0, 255), 2)
                    cv2.putText(img, '%d:%s' % (n, name), (pt[0] + 10, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(img, pt, 5, (0, 140, 255), -1)
            out = os.path.join(self.debug_dir, 'route.png')
            cv2.imwrite(out, img)
            self.get_logger().info('Saved annotated frame: %s' % out)
        except Exception as exc:
            self.get_logger().warn('Could not save debug image: %s' % exc)

    # ------------------------------------------------------------------ service
    def waypoint_callback(self, request, response):
        if not request.get_waypoints:
            self.get_logger().info('Request rejected (get_waypoints is False)')
            return response

        # A request can arrive before the frame has been processed: wait for the route to exist
        if not self.ready.wait(timeout=self.request_wait_s):
            self.get_logger().warn('Request arrived but no route is planned yet - returning an empty list')
            return response

        poses = []
        for (x, y, z) in self.waypoints:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = float(z)
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
