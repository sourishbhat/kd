#!/usr/bin/env python3
'''
Task 3A waypoint service.

From the single latched /image_raw frame it finds
    * critical survivors (red triangles), stable survivors (yellow circles), and the black obstacles,
    * marks every obstacle off-limits (convex hull + safety margin),
    * plans the CRITICAL route F6 -> 3 critical survivors (shortest order, around the obstacles),
and serves two services:
    'waypoints'    (GetWaypoints) : the critical route, as in Task 2C
    'plan_stable'  (PlanStable)   : the stable route, planned from the drone's position at the Switch,
                                    through the 3 stable survivors and back to F6.

marker_ids tag every waypoint so the client / server know what to do:
    'critical:<name>' / 'stable:<name>' -> survivor: Visit (Z=13, 1 s) then Save (Z=14, 2 s)
    'home:<name>'                        -> F6: hover (Z=13, 1 s), then land
    'via:<k>'                            -> intermediate point: fly straight past

Parameters (ros2 run swift_pico waypoint_service --ros-args -p obstacle_margin_wc:=0.45):
    obstacle_margin_wc  : clearance around each obstacle, in whycode units   (default 0.35)
    obstacle_v_max      : HSV value below which a pixel counts as obstacle    (default 45)
    obstacle_min_area   : smallest obstacle blob in rectified pixels          (default 2500)
    csv_path            : override path of the coordinates file (.json or .csv)
    debug_dir           : where annotated frames are saved (default ~/pico_ws/task3a_debug)
'''

import csv
import heapq
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
from waypoint_navigation.srv import GetWaypoints, PlanStable

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

START_INTERSECTION = 'F6'   # take-off and landing point
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


def rectify_frame(image):
    '''Rectified arena (RECT_SIZE x RECT_SIZE) from the four ArUco corner markers.
    Returns (warped, used_fallback). Falls back to "the frame is the arena" if the markers are unusable.'''
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_aruco_markers(gray)
        found = set(int(i) for i in ids) if ids is not None else set()
        missing = REQUIRED_MARKER_IDS - found
        if missing:
            raise ValueError('missing corner marker ID(s) %s' % sorted(missing))
        marker_corners = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids) if int(i) in REQUIRED_MARKER_IDS}
        return rectify_arena(image, get_ordered_arena_corners(marker_corners)), False
    except ValueError:
        return cv2.resize(image, (RECT_SIZE, RECT_SIZE)), True


# ============================================================================= obstacles
def find_obstacles(warped, v_max=45, min_area=2500, max_area=90000):
    '''Black obstacles = very dark blobs. Returns ([convex hulls], dark mask).'''
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] < v_max).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(c) for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    return hulls, mask


def detect_all(image, v_max=45, min_area=2500):
    '''Critical (red triangles), stable (yellow circles) and obstacles, all on the rectified frame.'''
    warped, used_fallback = rectify_frame(image)
    grid_points = build_grid_intersections()
    r_centers, r_cnts = find_color_centers(
        warped, [RED_LOWER_1, RED_LOWER_2], [RED_UPPER_1, RED_UPPER_2], shape_filter=is_triangle)
    y_centers, y_cnts = find_color_centers(warped, [YELLOW_LOWER], [YELLOW_UPPER], shape_filter=is_circle)
    hulls, dark_mask = find_obstacles(warped, v_max, min_area)
    return {
        'warped': warped,
        'fallback': used_fallback,
        'critical': nearest_intersection_names(r_centers, grid_points),
        'stable': nearest_intersection_names(y_centers, grid_points),
        'critical_contours': r_cnts,
        'stable_contours': y_cnts,
        'obstacles': hulls,
        'dark_mask': dark_mask,
    }


# ============================================================================= pixel <-> whycode
def fit_affine(px_grid, coords):
    '''Least-squares affine maps between rectified pixels and whycode (x, y) from every named intersection.
    Returns (M_p2w, M_w2p, max_residual_wc, px_per_wc). Use with apply_affine().'''
    names = [n for n in px_grid if n in coords]
    if len(names) < 3:
        raise ValueError('need at least 3 intersections present in both the grid and the coordinates file')
    P = np.array([[px_grid[n][0], px_grid[n][1], 1.0] for n in names])
    W = np.array([[coords[n][0], coords[n][1]] for n in names])
    W1 = np.hstack([W, np.ones((len(W), 1))])
    PX = np.array([[px_grid[n][0], px_grid[n][1]] for n in names])
    M_p2w = np.linalg.lstsq(P, W, rcond=None)[0]
    M_w2p = np.linalg.lstsq(W1, PX, rcond=None)[0]
    resid = float(np.abs(P @ M_p2w - W).max())
    px_per_wc = float(math.sqrt(abs(np.linalg.det(M_w2p[:2, :2]))))
    return M_p2w, M_w2p, resid, px_per_wc


def apply_affine(M, point):
    v = np.array([point[0], point[1], 1.0]) @ M
    return float(v[0]), float(v[1])


# ============================================================================= planning around obstacles
def path_length(points):
    return sum(math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
               for i in range(len(points) - 1))


class Planner:
    '''A* over a coarse grid of the rectified arena, with obstacles inflated by a safety margin, followed by
    line-of-sight smoothing so the route has only a few turning points.'''

    def __init__(self, hulls, px_per_wc, margin_wc=0.35, core_margin_wc=0.15, res=9, size=RECT_SIZE):
        self.size, self.res = size, res
        self.n = size // res
        core = np.zeros((size, size), np.uint8)
        for h in hulls:
            cv2.fillConvexPoly(core, h.reshape(-1, 2).astype(np.int32), 255)
        self.core = core

        def grow(px):
            k = int(round(px))
            if k <= 0:
                return core.copy()
            return cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))

        self.inflated = grow(margin_wc * px_per_wc)         # off-limits area
        self.core_safe = grow(core_margin_wc * px_per_wc)   # never entered, even near a survivor
        self.free_r = int(round(0.35 * px_per_wc))          # cleared around start/survivors/home
        self.blocked_px = self.inflated > 0
        self.cell_blocked = np.zeros((self.n, self.n), bool)

    def prepare(self, nodes_px):
        '''Survivors can sit close to an obstacle: clear the margin (not the obstacle itself) around nodes.'''
        blocked = self.inflated.copy()
        for (x, y) in nodes_px:
            cv2.circle(blocked, (int(x), int(y)), self.free_r, 0, -1)
        blocked = cv2.bitwise_or(blocked, self.core_safe)
        self.blocked_px = blocked > 0
        n, r = self.n, self.res
        self.cell_blocked = self.blocked_px.reshape(n, r, n, r).any(axis=(1, 3))

    def _cell(self, p):
        return (min(self.n - 1, max(0, int(p[0] // self.res))), min(self.n - 1, max(0, int(p[1] // self.res))))

    def _center(self, c):
        return ((c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res)

    def los(self, a, b):
        '''True if the straight segment a-b stays out of the blocked area.'''
        steps = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 3.0))
        for i in range(steps + 1):
            t = i / steps
            x = int(round(a[0] + (b[0] - a[0]) * t))
            y = int(round(a[1] + (b[1] - a[1]) * t))
            x = min(self.size - 1, max(0, x))
            y = min(self.size - 1, max(0, y))
            if self.blocked_px[y, x]:
                return False
        return True

    def astar(self, a, b):
        s, g = self._cell(a), self._cell(b)
        blocked, n = self.cell_blocked, self.n
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        counter = itertools.count()
        heap = [(0.0, next(counter), 0.0, s)]
        came, best = {}, {s: 0.0}
        while heap:
            _, _, gc, cur = heapq.heappop(heap)
            if cur == g:
                break
            if gc > best.get(cur, float('inf')):
                continue
            for dx, dy, cost in moves:
                nx, ny = cur[0] + dx, cur[1] + dy
                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                if blocked[ny, nx] and (nx, ny) != g and (nx, ny) != s:
                    continue
                if dx and dy and (blocked[cur[1], nx] or blocked[ny, cur[0]]):
                    continue  # no cutting corners between blocked cells
                ng = gc + cost
                if ng < best.get((nx, ny), float('inf')):
                    best[(nx, ny)] = ng
                    came[(nx, ny)] = cur
                    h = math.hypot(g[0] - nx, g[1] - ny)
                    heapq.heappush(heap, (ng + h, next(counter), ng, (nx, ny)))
        if g != s and g not in came:
            return None
        cells, cur = [g], g
        while cur != s:
            cur = came[cur]
            cells.append(cur)
        cells.reverse()
        return [self._center(c) for c in cells]

    def smooth(self, pts):
        out, i = [pts[0]], 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.los(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def path(self, a, b):
        '''Obstacle-free path a -> b in pixels (list of points, a first, b last), or None.'''
        if self.los(a, b):
            return [a, b]
        cells = self.astar(a, b)
        if cells is None:
            return None
        return self.smooth([a] + cells[1:-1] + [b])


def plan_route(planner, start, targets, end=None):
    '''start: (px, py). targets: [(name, (px, py))]. end: (name, (px, py)) or None.
    Tries every visiting order and keeps the shortest obstacle-free one.
    Returns [(kind, name, (px, py))] with kind in 'via' | 'target' | 'end'.'''
    nodes = [('start', start)] + list(targets) + ([end] if end else [])
    planner.prepare([p for _, p in nodes])
    cache = {}

    def leg(i, j):
        key = (min(i, j), max(i, j))
        if key not in cache:
            a, b = nodes[key[0]][1], nodes[key[1]][1]
            pth = planner.path(a, b)
            if pth is None:  # no way round found: straight line with a heavy penalty
                pth = [a, b]
                cache[key] = (pth, path_length(pth) * 10.0 + 1e6)
            else:
                cache[key] = (pth, path_length(pth))
        pth, cost = cache[key]
        return (pth if i <= j else pth[::-1]), cost

    k = len(targets)
    end_idx = [len(nodes) - 1] if end else []
    best_seq, best_cost = None, float('inf')
    for perm in itertools.permutations(range(1, k + 1)):
        seq = [0] + list(perm) + end_idx
        cost = sum(leg(seq[i], seq[i + 1])[1] for i in range(len(seq) - 1))
        if cost < best_cost:
            best_seq, best_cost = seq, cost

    route = []
    for i in range(len(best_seq) - 1):
        pth, _ = leg(best_seq[i], best_seq[i + 1])
        for p in pth[1:-1]:
            route.append(('via', None, p))
        dest = nodes[best_seq[i + 1]]
        is_end = bool(end) and best_seq[i + 1] == len(nodes) - 1
        route.append(('end' if is_end else 'target', dest[0], dest[1]))
    return route


def segments_to_waypoints(route, target_tag, coords, M_p2w):
    '''[(marker_id, x, y)]. Survivors / F6 use the exact coordinates from the grid file; via points use the
    pixel -> whycode fit.'''
    out, via = [], 0
    for kind, name, px in route:
        if kind == 'via':
            via += 1
            x, y = apply_affine(M_p2w, px)
            out.append(('via:%d' % via, x, y))
        else:
            tag = 'home' if kind == 'end' else target_tag
            out.append(('%s:%s' % (tag, name), coords[name][0], coords[name][1]))
    return out


def draw_debug(det, planner, routes, px_grid, path):
    '''Annotated frame: detections, off-limits areas and the planned route(s).'''
    img = det['warped'].copy()
    step = RECT_SIZE / GRID_CELLS
    for i in range(GRID_CELLS + 1):
        p = int(round(i * step))
        cv2.line(img, (p, 0), (p, RECT_SIZE - 1), (0, 255, 0), 1)
        cv2.line(img, (0, p), (RECT_SIZE - 1, p), (0, 255, 0), 1)

    over = img.copy()
    over[planner.inflated > 0] = (255, 0, 255)
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    cv2.drawContours(img, det['obstacles'], -1, (255, 255, 255), 2)
    cv2.drawContours(img, det['critical_contours'], -1, (0, 0, 255), 2)
    cv2.drawContours(img, det['stable_contours'], -1, (0, 255, 255), 2)

    for colour, route, start in routes:
        pts = [start] + [p for _, _, p in route]
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, colour, 3)
        for kind, name, p in route:
            if kind == 'via':
                cv2.circle(img, (int(p[0]), int(p[1])), 5, colour, -1)
            else:
                cv2.circle(img, (int(p[0]), int(p[1])), 12, colour, 2)
                cv2.putText(img, name, (int(p[0]) + 10, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (int(px_grid[START_INTERSECTION][0]), int(px_grid[START_INTERSECTION][1])), 8, (255, 0, 0), -1)
    cv2.imwrite(path, img)


# ============================================================================= ROS node
class WayPoints(Node):
    def __init__(self):
        super().__init__('waypoints_service')

        self.declare_parameter('csv_path', '')
        self.declare_parameter('debug_dir', os.path.expanduser('~/pico_ws/task3a_debug'))
        self.declare_parameter('request_wait_s', 25.0)
        self.declare_parameter('obstacle_margin_wc', 0.35)
        self.declare_parameter('obstacle_v_max', 45)
        self.declare_parameter('obstacle_min_area', 2500)

        self.debug_dir = self.get_parameter('debug_dir').value
        self.request_wait_s = float(self.get_parameter('request_wait_s').value)

        self.grid = self.load_coordinates()
        self.px_grid = {n: (x, y) for n, x, y in build_grid_intersections()}
        self.M_p2w, self.M_w2p, resid, self.px_per_wc = fit_affine(self.px_grid, self.grid)
        if resid > 0.1:
            self.get_logger().warn('Grid coordinates are not a clean affine map of the image (max residual %.2f '
                                   'whycode units) - via points may be off' % resid)

        self.critical_waypoints = []   # [(marker_id, x, y)]
        self.stable_names = []
        self.planner = None
        self.det = None
        self.ready = threading.Event()
        self.plan_lock = threading.Lock()

        group = ReentrantCallbackGroup()
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, qos,
                                                  callback_group=group)
        self.critical_srv = self.create_service(GetWaypoints, 'waypoints', self.waypoint_callback,
                                                callback_group=group)
        self.stable_srv = self.create_service(PlanStable, 'plan_stable', self.stable_callback,
                                              callback_group=group)
        self.get_logger().info('Waypoint service up. Waiting for /image_raw frame...')

    # ------------------------------------------------------------------ setup
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

    # ------------------------------------------------------------------ image -> critical route
    def image_callback(self, msg):
        try:
            image = image_msg_to_bgr(msg)
            det = detect_all(image, int(self.get_parameter('obstacle_v_max').value),
                             int(self.get_parameter('obstacle_min_area').value))
            if det['fallback']:
                self.get_logger().warn('Corner markers not found - assumed the whole frame is the arena')

            def usable(labels, what):
                out = []
                for n in labels:
                    if n in self.grid and n not in out:
                        out.append(n)
                    else:
                        self.get_logger().error('%s %s missing from the coordinates file - skipped' % (what, n))
                return out

            critical = usable(det['critical'], 'critical')
            stable = usable(det['stable'], 'stable')
            self.get_logger().info('Critical survivors: %s | Stable survivors: %s | Obstacles: %d' % (
                sorted(critical), sorted(stable), len(det['obstacles'])))
            if len(critical) != 3 or len(stable) != 3:
                self.get_logger().warn('Expected 3 critical + 3 stable survivors - check task3a_debug/')
            if not det['obstacles']:
                self.get_logger().warn('No obstacles found - check obstacle_v_max / obstacle_min_area')
            if not critical:
                raise RuntimeError('no critical survivors detected')

            self.det = det
            self.stable_names = stable
            self.planner = Planner(det['obstacles'], self.px_per_wc,
                                   float(self.get_parameter('obstacle_margin_wc').value))

            start = self.px_grid[START_INTERSECTION]
            with self.plan_lock:
                crit_route = plan_route(self.planner, start, [(n, self.px_grid[n]) for n in critical])
                crit_route_px = list(crit_route)
            self.critical_waypoints = segments_to_waypoints(crit_route, 'critical', self.grid, self.M_p2w)
            self.log_route('Critical route from %s' % START_INTERSECTION, self.critical_waypoints)

            # Demo of the stable route from F6, only for the debug image (the real one is planned at the Switch)
            routes = [((0, 140, 255), crit_route_px, start)]
            if stable:
                with self.plan_lock:
                    stable_route = plan_route(self.planner, start, [(n, self.px_grid[n]) for n in stable],
                                              (START_INTERSECTION, start))
                routes.append(((0, 255, 255), stable_route, start))
            os.makedirs(self.debug_dir, exist_ok=True)
            draw_debug(det, self.planner, routes, self.px_grid, os.path.join(self.debug_dir, 'plan.png'))
            self.get_logger().info('Saved annotated frame: %s' % os.path.join(self.debug_dir, 'plan.png'))
            self.ready.set()
        except Exception as exc:
            self.get_logger().error('Planning failed: %s' % exc)
        finally:
            self.destroy_subscription(self.image_sub)  # one frame is all we need

    def log_route(self, title, waypoints):
        self.get_logger().info('%s: %s' % (title, ' -> '.join(w[0] for w in waypoints)))
        for mid, x, y in waypoints:
            self.get_logger().info('  %-14s (%.2f, %.2f, %.2f)' % (mid, x, y, FLIGHT_Z))

    # ------------------------------------------------------------------ services
    def fill(self, response, waypoints):
        poses = []
        for _, x, y in waypoints:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = float(x), float(y), FLIGHT_Z
            pose.orientation.w = 1.0
            poses.append(pose)
        response.waypoints.header.frame_id = 'whycode'
        response.waypoints.header.stamp = self.get_clock().now().to_msg()
        response.waypoints.poses = poses
        response.marker_ids = [w[0] for w in waypoints]
        return response

    def waypoint_callback(self, request, response):
        if not request.get_waypoints:
            return response
        if not self.ready.wait(timeout=self.request_wait_s):
            self.get_logger().warn('GetWaypoints request but no route is planned yet - returning an empty list')
            return response
        self.get_logger().info('GetWaypoints: sending %d critical waypoints' % len(self.critical_waypoints))
        return self.fill(response, self.critical_waypoints)

    def stable_callback(self, request, response):
        if not self.ready.wait(timeout=self.request_wait_s):
            self.get_logger().warn('PlanStable request but no frame processed yet - returning an empty list')
            return response
        try:
            pos = (request.position.x, request.position.y)
            start = apply_affine(self.M_w2p, pos)
            start = (min(RECT_SIZE - 1, max(0.0, start[0])), min(RECT_SIZE - 1, max(0.0, start[1])))
            home = (START_INTERSECTION, self.px_grid[START_INTERSECTION])
            with self.plan_lock:
                route = plan_route(self.planner, start,
                                   [(n, self.px_grid[n]) for n in self.stable_names], home)
            waypoints = segments_to_waypoints(route, 'stable', self.grid, self.M_p2w)
            self.log_route('Stable route from (%.2f, %.2f)' % pos, waypoints)
            try:
                os.makedirs(self.debug_dir, exist_ok=True)
                draw_debug(self.det, self.planner, [((0, 255, 255), route, start)], self.px_grid,
                           os.path.join(self.debug_dir, 'stable_from_switch.png'))
            except Exception:
                pass
            return self.fill(response, waypoints)
        except Exception as exc:
            self.get_logger().error('Stable planning failed: %s' % exc)
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
