import cv2
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, LineString
import itertools
import math
from typing import Tuple, List, Optional
from utils import Point, Pose, ROBOT_RADIUS_CM
import time
import matplotlib.pyplot as plt


class VisionSystem:
    def __init__(
        self,
        camera_index: int = 1,
        camera_resolution: Tuple[int, int] = (1920, 1080),
        map_size_cm: Tuple[float, float] = (112, 82),
        warmup_time: int = 5,
    ):
        """
        Initializes the Vision System.
        Connects to the camera and sets up the perspective transformation matrix.

        :param camera_index: Index of the video capture (camera) device to use (default 0)
        :param camera_resolution: Desired camera resolution (width, height) in pixels
        :param map_size_cm: Size of the map in centimeters (width, height)
        """

        assert map_size_cm[0] > 0 and map_size_cm[1] > 0, "Map size must be positive non-zero values"

        self.map_width = map_size_cm[0]
        """Width of the real-world map in cm"""
        self.map_height = map_size_cm[1]
        """Height of the real-world map in cm"""

        # Init capture device and set resolution
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.release()
            raise RuntimeError(
                f"Could not open camera {camera_index}. Is it connected? Is another app using it?"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])

        self.warmup_camera(warmup_time)

        # Capture a frame for calibration
        ret, frame = self.cap.read()
        if not ret:
            self.release()
            raise RuntimeError("Could not capture frame for calibration")
        self.img = frame
        # Convert to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imwrite("warmup_frame.jpg", frame_rgb)

        # Detect markers
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, rejected = detector.detectMarkers(frame)

        self.src_points = self._detect_calibration_aruco(frame, corners, ids)
        # self.src_points = self._detect_calibration_markers(frame_rgb)

        self.map_width_pxl = int(self.src_points[1][0] - self.src_points[0][0])  # = TR.x - TL.x
        """Width of the real-world map in pixels (as seen by camera)"""
        self.map_height_pxl = int(self.src_points[2][1] - self.src_points[0][1])  # = BL.y - TL.y
        """Height of the real-world map in pixels (as seen by camera)"""

        self.pxl_per_cm_x = self.map_width_pxl / self.map_width
        self.pxl_per_cm_y = self.map_height_pxl / self.map_height

        self.dst_points = np.array(
            [
                [0, 0],
                [self.map_width_pxl, 0],
                [0, self.map_height_pxl],
                [self.map_width_pxl, self.map_height_pxl],
            ],
            dtype=np.float32,
        )

        self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # --- ArUco Setup ---
        # Using 4x4 dictionary, 50 markers (standard for small projects)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Use ArucoDetector if available (OpenCV 4.7+), otherwise fallback might be needed but we assume 4.7+
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def release(self):
        """Releases the camera resource."""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
            print("Camera released.")

    def __del__(self):
        self.release()

    def init_from_img(self, img: np.ndarray):
        """
        Helper "constructor" for testing (init with RGB img instead of a camera).

        Assumes VisionSystem instance was created with __new__ (instead of __init__) and `self.map_width` and `self.map_height` were set.
        """

        # Use provided image for calibration
        frame_rgb = img
        self.img = img

        # Detect markers
        self.src_points = self._detect_calibration_markers(frame_rgb)

        self.map_width_pxl = int(self.src_points[1][0] - self.src_points[0][0])  # = TR.x - TL.x
        self.map_height_pxl = int(self.src_points[2][1] - self.src_points[0][1])  # = BL.y - TL.y

        self.pxl_per_cm_x = self.map_width_pxl / self.map_width
        self.pxl_per_cm_y = self.map_height_pxl / self.map_height

        self.dst_points = np.array(
            [
                [0, 0],
                [self.map_width_pxl, 0],
                [0, self.map_height_pxl],
                [self.map_width_pxl, self.map_height_pxl],
            ],
            dtype=np.float32,
        )

        self.perspective_matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # --- ArUco Setup ---
        # Using 4x4 dictionary, 50 markers (standard for small projects)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        # Use ArucoDetector if available (OpenCV 4.7+), otherwise fallback might be needed but we assume 4.7+
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def warmup_camera(self, duration=5):
        # "Smart" Warmup: Wait for Light, not just Time

        print(f"Warming up camera for {duration}s or until exposure settles...")
        start_time = time.time()
        valid_frames = 0

        while True:
            ret, frame = self.cap.read()

            # Timeout check
            if time.time() - start_time > duration + 2.0:  # Give it a little grace period
                raise RuntimeError("Camera warmup timed out! (Received only black/empty frames)")

            if not ret or frame is None:
                time.sleep(0.1)
                continue

            # Check Brightness
            # If the image is pitch black (mean pixel value < 10), Auto-Exposure hasn't kicked in yet.
            # We discard these frames.
            mean_brightness = np.mean(frame)
            if mean_brightness < 10:
                continue

            # If we get here, the frame is valid and has light.
            # We consume a few more to ensure stability (e.g., 30 frames ~ 1 sec)
            valid_frames += 1
            if valid_frames > 30:
                break

        print(f"Camera ready. (Resolution: {self.cap.get(3)}x{self.cap.get(4)})")

    def _detect_calibration_aruco(self, img, corners, ids) -> np.ndarray:
        corners_centers = []
        for i, c in enumerate(corners):
            marker_id = ids[i][0]  # get scalar I
            if marker_id != 1:
                pts = c[0]
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()
                corners_centers.append([marker_id, [cx, cy]])  # (id, [x, y])

        # Sort by ID
        corners_centers.sort(key=lambda x: x[0])
        corners_centers = np.array([item[1] for item in corners_centers])

        if len(corners_centers) != 4:
            raise RuntimeError(f"Calibration failed: Expected 4 markers, found {len(corners_centers)}")
        return corners_centers

    def _detect_calibration_markers(self, img: np.ndarray) -> np.ndarray:
        """
        Detects the 4 green calibration markers in the `img` corners.

        :returns: sorted list of the 4 corner points: [TL, TR, BL, BR]
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Green color range (adjust if needed)
        lower_green = np.array([50, 50, 50])
        upper_green = np.array([100, 100, 120])

        mask = cv2.inRange(hsv, lower_green, upper_green)
        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for cnt in contours:
            # Filter by area to avoid noise
            area = cv2.contourArea(cnt)
            if area > 10:  # Minimum area threshold
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    centroids.append([cX, cY])

        if len(centroids) != 4:
            raise RuntimeError(f"Calibration failed: Expected 4 markers, found {len(centroids)}")

        centroids = np.array(centroids, dtype=np.float32)

        # Sort points to match [TL, TR, BL, BR] order
        # 1. Sort by Y to separate Top (2 points) and Bottom (2 points)
        centroids = centroids[np.argsort(centroids[:, 1])]

        top = centroids[:2]
        bottom = centroids[2:]

        # 2. Sort Top by X to get TL, TR
        top = top[np.argsort(top[:, 0])]

        # 3. Sort Bottom by X to get BL, BR
        bottom = bottom[np.argsort(bottom[:, 0])]
        return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures a single frame from the camera and converts it to RGB."""

        # print("Capturing frame from camera...")
        ret, frame = self.cap.read()
        # print(f"Frame captured: {ret}, shape: {frame.shape if ret else 'N/A'}")
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def warp_image(self, img: np.ndarray) -> np.ndarray:
        """Applies the perspective transformation to the image."""

        return cv2.warpPerspective(img, self.perspective_matrix, (self.map_width_pxl, self.map_height_pxl))

    def get_robot_pose(self) -> Optional[Pose]:
        """
        Captures an image, warps it, and detects the robot's pose using ArUco markers.

        :returns: `Pose(x, y, theta)` in cm and radians.
        """

        img = self.capture_frame()
        if img is None:
            return None

        warped = self.warp_image(img)
        self.latest_warped_img = warped
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

        # Detect ArUco markers on the warped, grayscale image
        all_aruco_corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Assume the first marker found is the robot
            robot_aruco_corners = all_aruco_corners[0][0]  # Shape (4, 2), ie 4x (x, y) points

            # Flip y-axis to be y-up
            robot_aruco_corners[:, 1] = self.map_height_pxl - robot_aruco_corners[:, 1]

            # Compute center
            cx = np.mean(robot_aruco_corners[:, 0])
            cy = np.mean(robot_aruco_corners[:, 1])

            # Convert pixels to cm
            x_cm = cx / self.pxl_per_cm_x
            y_cm = cy / self.pxl_per_cm_y

            # Now compute the orientation (using left edge which corresponds to the robot's forward direction)
            forward_vec = robot_aruco_corners[0] - robot_aruco_corners[3]  # TL - BL
            theta = math.atan2(forward_vec[1], forward_vec[0])  # atan2(dy, dx)

            return Pose(x_cm, y_cm, theta)

        return None

    def construct_map(self, cspace_padding: float = 2.0) -> Tuple[nx.Graph, int, int]:
        """
        Full pipeline to construct the map's visibility graph from camera input.

        *Capture -> Warp -> Segment -> Polygonize -> Buffer -> Visibility Graph*

        :param cspace_padding: Additional padding (in cm) around obstacles for the robot's C-space.
        :returns: (Graph, start_node_id, goal_node_id).
        """

        img = self.img
        if img is None:
            img = self.capture_frame()
        if img is None:
            raise RuntimeError("Could not capture image from camera")

        warped = self.warp_image(img)
        cv2.imwrite("PPPPP.jpg", warped)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

        # 1. Segment Obstacles
        # Lower red (0-25 covers red through distinct orange)
        obstacle_lower1 = np.array([0, 70, 90])
        obstacle_upper1 = np.array([25, 255, 255])
        # Upper red (155-179 covers darker/purplish reds)
        obstacle_lower2 = np.array([155, 70, 90])
        obstacle_upper2 = np.array([179, 255, 255])

        obstacle_mask1 = cv2.inRange(hsv, obstacle_lower1, obstacle_upper1)
        obstacle_mask2 = cv2.inRange(hsv, obstacle_lower2, obstacle_upper2)

        obstacle_mask = cv2.bitwise_or(obstacle_mask1, obstacle_mask2)

        # Morphological operations to clean up noise
        # Increased kernel size slightly to fill larger holes
        kernel = np.ones((7, 7), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite("obstacle_mask.jpg", obstacle_mask)
        cv2.imwrite("obstacle_mask1.jpg", obstacle_mask1)
        cv2.imwrite("obstacle_mask2.jpg", obstacle_mask2)

        # 2. Segment Target (Goal)
        lower_green = np.array([35, 60, 80])
        upper_green = np.array([85, 255, 240])
        # white_mask = cv2.inRange(hsv, lower_red2, upper_green)
        target_mask = cv2.inRange(hsv, lower_green, upper_green)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("RRRRR.jpg", target_mask)
        target_indices = np.argwhere(target_mask > 0)
        if len(target_indices) == 0:
            raise RuntimeError("Target not detected")
        target_pos_px = np.mean(target_indices, axis=0)[::-1]  # (y, x) -> (x, y)
        goal_pos_cm = Point(
            target_pos_px[0] / self.pxl_per_cm_x, self.map_height - (target_pos_px[1] / self.pxl_per_cm_y)
        )

        # 3. Get Robot Pose (Start)
        # Try ArUco first by calling our own method (but we need to reuse the image ideally,
        # however get_robot_pose captures a new one. For simplicity, let's re-detect on current image)
        # Re-implementing ArUco detection on *this* frame to avoid movement issues
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        start_pos_cm: Optional[Point] = None

        if ids is not None and len(ids) > 0:
            c = corners[0][0]
            center_x = np.mean(c[:, 0])
            center_y = np.mean(c[:, 1])
            start_pos_cm = Point(
                center_x / self.pxl_per_cm_x, self.map_height - (center_y / self.pxl_per_cm_y)
            )
            print(start_pos_cm)
        else:
            # Fallback to color segmentation if ArUco fails
            robot_lower = np.array([0, 0, 220])
            robot_upper = np.array([179, 80, 255])
            robot_mask = cv2.inRange(hsv, robot_lower, robot_upper)
            robot_mask = cv2.morphologyEx(robot_mask, cv2.MORPH_OPEN, kernel)
            robot_mask = cv2.morphologyEx(robot_mask, cv2.MORPH_CLOSE, kernel)

            robot_indices = np.argwhere(robot_mask > 0)
            if len(robot_indices) == 0:
                raise RuntimeError("Robot not detected (ArUco or Color)")
            robot_pos_px = np.mean(robot_indices, axis=0)[::-1]
            start_pos_cm = Point(
                robot_pos_px[0] / self.pxl_per_cm_x, self.map_height - (robot_pos_px[1] / self.pxl_per_cm_y)
            )

        # 4. Polygon Approximation & Buffering
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_polygons = []
        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            poly_pts = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
            # Convert to cm immediately
            poly_pts = poly_pts.astype(float)
            poly_pts[:, 0] = poly_pts[:, 0] / self.pxl_per_cm_x
            poly_pts[:, 1] = self.map_height - (poly_pts[:, 1] / self.pxl_per_cm_y)

            polygon = Polygon(poly_pts)

            # Enlarge polygons by robot radius + padding, using a mitre buffer
            buffered_poly = polygon.buffer(
                ROBOT_RADIUS_CM + cspace_padding, join_style="mitre", mitre_limit=1.2, resolution=1
            )
            buffered_poly_pts = np.array(buffered_poly.exterior.coords).reshape(-1, 2)

            # Merge close vertices
            final_poly_pts = self._merge_close_vertices(buffered_poly_pts, min_dist=6.0)
            final_polygons.append(final_poly_pts)

        # 5. Construct Visibility Graph
        shapely_polygons = [Polygon(poly_pts) for poly_pts in final_polygons]
        G, _ = self._construct_visibility_graph(shapely_polygons, start_pos_cm, goal_pos_cm)

        # Identify start and goal IDs
        # In _construct_visibility_graph, we append start then goal to nodes
        start_node_id = G.number_of_nodes() - 2
        goal_node_id = G.number_of_nodes() - 1

        return G, start_node_id, goal_node_id

    def live_visu(self, g, start, end, waypoints, resolution: Tuple[int, int] = (1000, 600)):
        """
        Init live visualization
        """

        self.visu_window_name = "Live Map"
        cv2.namedWindow(self.visu_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.visu_window_name, *resolution)

        # Store map elements for redrawing on live frames
        self.visu_g = g
        self.visu_start = start
        self.visu_end = end
        self.visu_waypoints = waypoints

        # Create a fallback background image
        self.bg_img = self.warp_image(self.img).copy()

    def update_robot_visu(self, robot_pose: Optional[Pose] = None, target: Optional[Point] = None) -> bool:
        """
        Update robot marker on live visualization.
        :param robot_pose: Current robot pose (optional, will capture if None)
        :param target: Current target point (optional, for heading visualization)
        :return: True if 'q' was pressed to quit, False otherwise
        """
        if robot_pose is None:
            robot_pose = self.get_robot_pose()

        # Use latest warped image if available, else static bg
        if hasattr(self, "latest_warped_img"):
            # Convert RGB to BGR for OpenCV display
            frame = cv2.cvtColor(self.latest_warped_img, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(self.bg_img, cv2.COLOR_RGB2BGR)

        H = self.map_height

        # --- Draw Map Elements (Graph, Path) ---
        if hasattr(self, "visu_g"):
            # Draw edges
            for u, v in self.visu_g.edges():
                p1 = self.visu_g.nodes[u]["pos"]
                p2 = self.visu_g.nodes[v]["pos"]
                pt1 = (int(p1[0] * self.pxl_per_cm_x), int((H - p1[1]) * self.pxl_per_cm_y))
                pt2 = (int(p2[0] * self.pxl_per_cm_x), int((H - p2[1]) * self.pxl_per_cm_y))
                cv2.line(frame, pt1, pt2, (200, 200, 255), 1)  # Light Red

            # Draw Path
            if self.visu_waypoints:
                for i in range(len(self.visu_waypoints) - 1):
                    p1 = self.visu_waypoints[i]
                    p2 = self.visu_waypoints[i + 1]
                    pt1 = (int(p1[0] * self.pxl_per_cm_x), int((H - p1[1]) * self.pxl_per_cm_y))
                    pt2 = (int(p2[0] * self.pxl_per_cm_x), int((H - p2[1]) * self.pxl_per_cm_y))
                    cv2.line(frame, pt1, pt2, (255, 255, 0), 2)  # Cyan

            # Draw Start/Goal
            start_pos = self.visu_g.nodes[self.visu_start]["pos"]
            goal_pos = self.visu_g.nodes[self.visu_end]["pos"]
            s_pt = (int(start_pos[0] * self.pxl_per_cm_x), int((H - start_pos[1]) * self.pxl_per_cm_y))
            g_pt = (int(goal_pos[0] * self.pxl_per_cm_x), int((H - goal_pos[1]) * self.pxl_per_cm_y))
            cv2.circle(frame, s_pt, 8, (0, 255, 0), -1)
            cv2.circle(frame, g_pt, 8, (0, 0, 255), -1)

        # --- Draw Robot Pose ---
        if robot_pose:
            rx, ry, rtheta = robot_pose
            # Robot Center
            cx = int(rx * self.pxl_per_cm_x)
            cy = int((H - ry) * self.pxl_per_cm_y)

            # Direction Vector (Length 30px)
            # World: x=cos, y=sin. Image: x=cos, y=-sin (because y is flipped)
            dx = int(30 * math.cos(rtheta))
            dy = int(30 * -math.sin(rtheta))

            cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (150, 20, 255), 2, tipLength=0.5)
            cv2.circle(frame, (cx, cy), 5, (150, 20, 255), -1)

            # Text Label
            pos_text = f"x={rx:.1f} y={ry:.1f}"
            cv2.putText(
                frame,
                pos_text,
                (cx + 12, cy - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (150, 20, 255),
                2,
                cv2.LINE_AA,
            )
            orient_text = f"theta={math.degrees(rtheta):.1f}"
            cv2.putText(
                frame,
                orient_text,
                (cx + 12, cy + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (150, 20, 255),
                2,
                cv2.LINE_AA,
            )

            # --- Draw Controller Info (Target & Heading Error) ---
            if target:
                tx, ty = target
                tx_px = int(tx * self.pxl_per_cm_x)
                ty_px = int((H - ty) * self.pxl_per_cm_y)

                # Line to target
                cv2.line(frame, (cx, cy), (tx_px, ty_px), (0, 255, 255), 1)  # Yellow

                # Calculate heading error for display
                angle_to_target = math.atan2(ty - ry, tx - rx)
                heading_error = (angle_to_target - rtheta + math.pi) % (2 * math.pi) - math.pi

                err_text = f"Err: {math.degrees(heading_error):.1f} deg"
                cv2.putText(
                    frame,
                    err_text,
                    (cx + 10, cy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow(self.visu_window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return True
        return False

    def _merge_close_vertices(self, pts: np.ndarray, min_dist: float = 5.0) -> np.ndarray:
        """
        Merges vertices in a polygon that are closer than `min_dist` cm apart.

        :param pts: np.ndarray of shape (N, 2) representing polygon vertices in world space (cm).
        :param min_dist: Minimum distance needed between vertices (cm) to avoid merging.
        :returns: np.ndarray of shape (M, 2) with merged vertices, where M <= N.
        """

        if len(pts) <= 2:
            return pts

        # Go through each pair of adjacent points, merging those that are too close to each other
        new_pts = [pts[0]]
        for i in range(1, len(pts)):
            dist = np.linalg.norm(pts[i] - new_pts[-1])
            if dist > min_dist:
                new_pts.append(pts[i])
            else:
                new_pts[-1] = 0.5 * (new_pts[-1] + pts[i])  # Merged point is the mean

        # Check distance between last and first point (closed polygon)
        if np.linalg.norm(np.array(new_pts[-1]) - np.array(new_pts[0])) < min_dist:
            new_pts.pop()

        return np.array(new_pts).reshape(-1, 2)

    def _construct_visibility_graph(
        self,
        obstacles: List[Polygon],
        start_pos: Point,
        goal_pos: Point,
    ) -> Tuple[nx.Graph, List[Point]]:
        """
        Constructs a visibility graph given polygon obstacles, and start & goal positions.

        :param obstacles: List of Shapely Polygons representing the obstacles in world space (cm).
        :param start_pos: Starting point in world space (cm).
        :param goal_pos: Goal point in world space (cm).
        """

        G = nx.Graph()
        # Collect all nodes (pts): obstacle vertices + start + goal
        obstacle_vertices = [Point(*pt) for poly in obstacles for pt in poly.exterior.coords[:-1]]
        all_pts = obstacle_vertices + [start_pos, goal_pos]

        # Add all nodes to graph - list[Point] -> {index: {"pos": Point}}
        for i, pt in enumerate(all_pts):
            G.add_node(i, pos=pt)

        # Add the obstacle edges first
        for poly in obstacles:
            poly_pts = list(poly.exterior.coords)[:-1]  # Exclude last point (same as first)
            num_pts = len(poly_pts)
            for i in range(num_pts):
                pt1 = Point(*poly_pts[i])
                pt2 = Point(*poly_pts[(i + 1) % num_pts])

                # Find index in all_pts
                idx1 = self._find_point_index(pt1, all_pts)
                idx2 = self._find_point_index(pt2, all_pts)

                if idx1 is not None and idx2 is not None:
                    dist = math.dist(pt1, pt2)
                    G.add_edge(idx1, idx2, weight=dist)

        # Visibility edges
        for (i, pt1), (j, pt2) in itertools.combinations(enumerate(all_pts), 2):
            if i == j:
                continue

            # Skip if edge already exists (obstacle edge)
            if G.has_edge(i, j):
                continue

            line = LineString([pt1, pt2])
            dist = line.length

            # Skip if line is invalid ("zero" length)
            if dist < 1e-6:
                continue

            # --- INTERSECTION TEST ---
            is_visible = True

            # We shrink the line slightly to avoid "touching" errors at vertices.
            # Done by interpolating 0.1% from start and 99.9% from start (=0.1% from end).
            l_start = line.interpolate(0.001 * dist)
            l_end = line.interpolate(0.999 * dist)
            test_line = LineString([l_start, l_end])

            for poly in obstacles:
                # Check 1: Does the shrunk line hit an obstacle?
                if test_line.intersects(poly):
                    is_visible = False
                    break

                # Check 2 (Crucial for Concave shapes):
                # If the line is completely WITHIN a single polygon (e.g. connecting tips of a 'U' shape),
                # intersects might be False, but the path is invalid.
                if test_line.within(poly):
                    is_visible = False
                    break

            # 3. Add edge if visible
            if is_visible:
                G.add_edge(i, j, weight=dist)

        return G, all_pts

    def _find_point_index(self, pt: Point, all_pts: List[Point]) -> Optional[int]:
        """Finds the index of a Point in a list of Points, using approximate equality."""

        for i, pt2 in enumerate(all_pts):
            if math.isclose(pt.x, pt2.x, abs_tol=1e-4) and math.isclose(pt.y, pt2.y, abs_tol=1e-4):
                return i
        return None
