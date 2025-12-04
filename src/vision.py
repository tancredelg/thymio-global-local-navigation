import time
import math
import itertools
from typing import Tuple, List, Optional

import cv2
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, LineString

from utils import Point, Pose, ROBOT_RADIUS_CM, RobotState, MissionState


class VisionSystem:
    """
    Handles all computer vision tasks for the Thymio navigation system.

    Responsibilities:
    - Camera connection and calibration (perspective warp).
    - Map construction (obstacle segmentation, visibility graph).
    - Robot localization (ArUco markers).
    - Live visualization of the system state.
    """

    def __init__(
        self,
        camera_index: int = 1,
        camera_resolution: Tuple[int, int] = (1920, 1080),
        map_size_cm: Tuple[float, float] = (132.5, 92.5),
        warmup_time: int = 5,
    ):
        """
        Initializes the Vision System.
        Connects to the camera and sets up the perspective transformation matrix.

        :param camera_index: Index of the video capture (camera) device to use.
        :param camera_resolution: Desired camera resolution (width, height) in pixels.
        :param map_size_cm: Size of the real-world map in centimeters (width, height).
        :param warmup_time: Duration in seconds to warm up the camera auto-exposure.
        """
        assert map_size_cm[0] > 0 and map_size_cm[1] > 0, "Map size must be positive."

        self.map_width = map_size_cm[0]
        """Width of the real-world map in cm."""
        self.map_height = map_size_cm[1]
        """Height of the real-world map in cm."""

        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.release()
            raise RuntimeError(
                f"Could not open camera {camera_index}. Is it connected? Is another app using it?"
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])

        self.warmup_camera(warmup_time)

        # --- Calibration ---
        # Capture a frame to detect the 4 corner markers for perspective warping
        ret, frame = self.cap.read()
        if not ret:
            self.release()
            raise RuntimeError("Could not capture frame for calibration.")

        self.img = frame  # Store initial frame

        # Detect ArUco markers for calibration
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector = cv2.aruco.ArucoDetector(aruco_dict)
        corners, ids, _ = detector.detectMarkers(frame)

        # Calculate source points (corners of the map in the image)
        self.src_points = self._detect_calibration_aruco(self.img, corners, ids)

        # Calculate destination points (corners of the map in the warped image)
        self.map_width_pxl = int(self.src_points[1][0] - self.src_points[0][0])  # TR.x - TL.x
        """Width of the warped map image in pixels."""
        self.map_height_pxl = int(self.src_points[2][1] - self.src_points[0][1])  # BL.y - TL.y
        """Height of the warped map image in pixels."""

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

        # --- Runtime ArUco Setup ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def release(self):
        """Releases the camera resource."""
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
            print("Camera released.")

    def __del__(self):
        self.release()

    # =========================================================================
    # Camera Operations
    # =========================================================================

    def warmup_camera(self, duration: int = 5):
        """
        Warms up the camera by discarding frames until auto-exposure settles.

        :param duration: Maximum duration to wait in seconds.
        """
        print(f"Warming up camera for {duration}s or until exposure settles...")
        start_time = time.time()
        valid_frames = 0

        while True:
            ret, frame = self.cap.read()

            # Timeout check
            if time.time() - start_time > duration + 2.0:
                raise RuntimeError("Camera warmup timed out! (Received only black/empty frames)")

            if not ret or frame is None:
                time.sleep(0.1)
                continue

            # Check Brightness: If image is pitch black, AE hasn't kicked in.
            if np.mean(frame) < 10:
                continue

            # Consume a few valid frames to ensure stability
            valid_frames += 1
            if valid_frames > 30:
                break

        print(f"Camera ready. (Resolution: {self.cap.get(3)}x{self.cap.get(4)})")

    def capture_frame(self) -> Optional[np.ndarray]:
        """Captures a single frame from the camera and converts it to RGB."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def warp_image(self, img: np.ndarray) -> np.ndarray:
        """Applies the perspective transformation to the image."""
        return cv2.warpPerspective(img, self.perspective_matrix, (self.map_width_pxl, self.map_height_pxl))

    # =========================================================================
    # Localization
    # =========================================================================

    def get_robot_pose(self) -> Optional[Pose]:
        """
        Captures an image, warps it, and detects the robot's pose using ArUco markers.

        :returns: `Pose(x, y, theta)` in cm and radians, or None if not found.
        """
        img = self.capture_frame()
        if img is None:
            return None

        warped = self.warp_image(img)
        self.latest_warped_img = warped
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)

        # Detect ArUco markers
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Assume the first marker found is the robot
            # Shape is (1, 4, 2) -> (4, 2)
            robot_corners = corners[0][0]

            # Flip y-axis to match world coordinates (Y-up)
            robot_corners[:, 1] = self.map_height_pxl - robot_corners[:, 1]

            # Compute center
            cx = np.mean(robot_corners[:, 0])
            cy = np.mean(robot_corners[:, 1])

            # Convert pixels to cm
            x_cm = cx / self.pxl_per_cm_x
            y_cm = cy / self.pxl_per_cm_y

            # Compute orientation using the top-left and bottom-left corners
            # (Corresponds to the robot's forward direction in standard ArUco orientation)
            forward_vec = robot_corners[0] - robot_corners[3]  # TL - BL
            theta = math.atan2(forward_vec[1], forward_vec[0])

            return Pose(x_cm, y_cm, theta)

        return None

    # =========================================================================
    # Mapping & Path Planning
    # =========================================================================

    def build_static_map(self, cspace_padding: float = 2.0) -> None:
        """
        Constructs the static map (obstacles + goal) from camera input.
        Does NOT include the robot start position.

        Pipeline: Capture -> Warp -> Segment -> Polygonize -> Buffer -> Visibility Graph

        :param cspace_padding: Additional padding (in cm) around obstacles for the robot's C-space.
        """
        img = self.img
        if img is None:
            img = self.capture_frame()
        if img is None:
            raise RuntimeError("Could not capture image from camera")

        warped = self.warp_image(img)
        cv2.imwrite("../vision_debug/warped.jpg", warped)

        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (7, 7), 0)

        # --- 1. Segment Obstacles ---
        # Dark blue-ish obstacles
        lo_blue = np.array([90, 40, 80])
        hi_blue = np.array([165, 255, 255])
        obstacle_mask = cv2.inRange(hsv, lo_blue, hi_blue)

        # Morphological operations to clean up noise
        kernel = np.ones((7, 7), np.uint8)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_OPEN, kernel)
        obstacle_mask = cv2.morphologyEx(obstacle_mask, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("../vision_debug/obstacle_mask.jpg", obstacle_mask)

        # --- 2. Segment Target (Goal) ---
        # Red heart target
        # Lower Red (covers 0-15 deg)
        lo_red1 = np.array([0, 50, 50])
        hi_red1 = np.array([15, 255, 255])
        # Upper Red (covers 165-180 deg)
        lo_red2 = np.array([165, 50, 50])
        hi_red2 = np.array([180, 255, 255])

        target_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lo_red1, hi_red1),
            cv2.inRange(hsv, lo_red2, hi_red2),
        )

        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)
        target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite("../vision_debug/target_mask.jpg", target_mask)

        target_indices = np.argwhere(target_mask > 0)
        if len(target_indices) == 0:
            raise RuntimeError("Target not detected")

        target_pos_px = np.mean(target_indices, axis=0)[::-1]  # (y, x) -> (x, y)
        goal_pos_cm = Point(
            target_pos_px[0] / self.pxl_per_cm_x, self.map_height - (target_pos_px[1] / self.pxl_per_cm_y)
        )

        # --- 3. Polygon Approximation & Buffering ---
        contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final_polygons = []
        for cnt in contours:
            epsilon = 0.015 * cv2.arcLength(cnt, True)
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

        # --- 4. Construct Base Visibility Graph (Obstacles + Goal) ---
        self.shapely_polygons = [Polygon(poly_pts) for poly_pts in final_polygons]
        self.base_graph, _ = self._construct_visibility_graph(self.shapely_polygons, None, goal_pos_cm)

        # Identify goal ID (largest node ID)
        self.goal_node_idx = max(self.base_graph.nodes)

        print(f"Static map built. Base graph has {self.base_graph.number_of_nodes()} nodes.")

    def add_robot_to_graph(self, robot_pose: Pose) -> Tuple[nx.Graph, int, int]:
        """
        Augments the base visibility graph with the robot's current position as the start node.

        :param robot_pose: Current robot pose.
        :return: (Augmented Graph, start_node_idx, goal_node_idx)
        """
        if not hasattr(self, "base_graph"):
            raise RuntimeError("Base graph not built. Call build_static_map() first.")

        # Create a copy of the base graph
        G = self.base_graph.copy()

        start_pos = Point(robot_pose.x, robot_pose.y)
        start_node_idx = max(G.nodes) + 1  # Next available ID (largest ID + 1)

        # Add start node
        G.add_node(start_node_idx, pos=start_pos)

        # Connect start node to all other visible nodes
        for node_idx in G.nodes:
            if node_idx == start_node_idx:
                continue

            target_pos = G.nodes[node_idx]["pos"]

            if self._is_visible(start_pos, target_pos, self.shapely_polygons):
                dist = math.dist(start_pos, target_pos)
                G.add_edge(start_node_idx, node_idx, weight=dist)

        return G, start_node_idx, self.goal_node_idx

    # =========================================================================
    # Visualization
    # =========================================================================

    def init_visu(
        self,
        g: nx.Graph,
        start_idx: int,
        end_idx: int,
        waypoints: List[Point],
        resolution: Tuple[int, int] = (1300, 910),
    ):
        """Initializes the live visualization window."""
        self.visu_window_name = "Live Map"
        cv2.namedWindow(self.visu_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.visu_window_name, *resolution)

        # Store map elements for redrawing on live frames
        self.visu_g = g
        self.visu_start_node_idx = start_idx
        self.visu_end_node_idx = end_idx
        self.visu_waypoints = waypoints

        # Create a fallback background image
        self.visu_bg_img = self.warp_image(self.img).copy()

        self.reset_trajectories()

    def reset_trajectories(self):
        """Clear all stored trajectories for visualization."""
        self.ekf_est_traj = []
        self.ekf_pred_traj = []

    def update_robot_visu(
        self,
        mission_state: MissionState,
        controller_state: RobotState,
        robot_pose: Optional[Pose],
        target_point: Optional[Point] = None,
        ekf_pred_pose: Optional[Pose] = None,
        ekf_cov: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Update robot marker on live visualization.

        :param mission_state: High-level mission state (e.g., RUNNING / FINISHED)
        :param controller_state: Low-level controller state (RobotState)
        :param robot_pose: Current robot Pose from vision (can be None)
        :param target: Current target point (optional, for heading visualization)
        :param ekf_pred_pose: Predicted Pose from EKF (before update)
        :param ekf_cov: Covariance matrix 3x3 from EKF (for ellipse)
        :return: True if 'q' is pressed to exit, False otherwise
        """
        # Initialize trajectory buffers if they don't exist
        if not hasattr(self, "ekf_est_traj"):
            self.ekf_est_traj = []  # list (x, y) in cm (estimation from EKF)
        if not hasattr(self, "ekf_pred_traj"):
            self.ekf_pred_traj = []  # list (x, y) in cm (predicted from EKF)

        # Use latest warped image if available, else static bg
        if hasattr(self, "latest_warped_img"):
            frame = cv2.cvtColor(self.latest_warped_img, cv2.COLOR_RGB2BGR)
        else:
            frame = cv2.cvtColor(self.visu_bg_img, cv2.COLOR_RGB2BGR)

        H = self.map_height

        # --- Update Trajectories ---
        if mission_state == MissionState.RUNNING and robot_pose is not None:
            self.ekf_est_traj.append((float(robot_pose.x), float(robot_pose.y)))

        if mission_state == MissionState.RUNNING and ekf_pred_pose is not None:
            self.ekf_pred_traj.append((float(ekf_pred_pose.x), float(ekf_pred_pose.y)))

        # --- Draw Elements ---
        self._draw_map_elements(frame, H)

        if ekf_pred_pose is not None:
            self._draw_predicted_pose(frame, ekf_pred_pose, H)

        if mission_state == MissionState.RUNNING and robot_pose is not None and ekf_cov is not None:
            self._draw_uncertainty_ellipse(frame, robot_pose, ekf_cov, H)

        if robot_pose:
            self._draw_robot(frame, robot_pose, target_point, H)

        self._draw_trajectories(frame, H)
        self._draw_state_text(frame, mission_state, controller_state)

        cv2.imshow(self.visu_window_name, frame)
        # cv2.imwrite("../vision_debug/visu.jpg", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            return True
        return False

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def _draw_map_elements(self, frame: np.ndarray, H: float):
        """Draws graph edges, nodes, path, and start/goal points."""
        if not hasattr(self, "visu_g"):
            return

        edge_colour = (255, 75, 0)  # Light Red
        node_colour = (255, 150, 20)  # Orange
        path_colour = (100, 200, 0)  # Cyan
        start_colour = (100, 200, 0)  # Cyan
        goal_colour = (0, 0, 180)  # Red

        # Draw edges
        for u, v in self.visu_g.edges():
            p1 = self.visu_g.nodes[u]["pos"]
            p2 = self.visu_g.nodes[v]["pos"]
            pt1 = (int(p1[0] * self.pxl_per_cm_x), int((H - p1[1]) * self.pxl_per_cm_y))
            pt2 = (int(p2[0] * self.pxl_per_cm_x), int((H - p2[1]) * self.pxl_per_cm_y))
            cv2.line(frame, pt1, pt2, edge_colour, 1)

        # Draw nodes (dots)
        for n in self.visu_g.nodes():
            px, py = self.visu_g.nodes[n]["pos"]
            pt = (int(px * self.pxl_per_cm_x), int((H - py) * self.pxl_per_cm_y))
            cv2.circle(frame, pt, 4, node_colour, -1)

        # Draw Path
        if self.visu_waypoints:
            for i in range(len(self.visu_waypoints) - 1):
                p1 = self.visu_waypoints[i]
                p2 = self.visu_waypoints[i + 1]
                pt1 = (int(p1[0] * self.pxl_per_cm_x), int((H - p1[1]) * self.pxl_per_cm_y))
                pt2 = (int(p2[0] * self.pxl_per_cm_x), int((H - p2[1]) * self.pxl_per_cm_y))
                cv2.line(frame, pt1, pt2, path_colour, 2)

        # Draw Start/Goal
        start_pos = self.visu_g.nodes[self.visu_start_node_idx]["pos"]
        goal_pos = self.visu_g.nodes[self.visu_end_node_idx]["pos"]
        s_pt = (int(start_pos[0] * self.pxl_per_cm_x), int((H - start_pos[1]) * self.pxl_per_cm_y))
        g_pt = (int(goal_pos[0] * self.pxl_per_cm_x), int((H - goal_pos[1]) * self.pxl_per_cm_y))
        cv2.circle(frame, s_pt, 8, start_colour, -1)
        cv2.circle(frame, g_pt, 8, goal_colour, -1)

    def _draw_predicted_pose(self, frame: np.ndarray, ekf_pred_pose: Pose, H: float):
        """Draws the predicted pose from EKF as a yellow dot."""
        px = int(ekf_pred_pose.x * self.pxl_per_cm_x)
        py = int((H - ekf_pred_pose.y) * self.pxl_per_cm_y)
        cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

    def _draw_uncertainty_ellipse(self, frame: np.ndarray, robot_pose: Pose, ekf_cov: np.ndarray, H: float):
        """Draws the uncertainty ellipse from the EKF covariance matrix."""
        try:
            # Extract position covariance (x, y)
            Sigma_xy = np.array(
                [
                    [ekf_cov[0, 0], ekf_cov[0, 1]],
                    [ekf_cov[1, 0], ekf_cov[1, 1]],
                ],
                dtype=float,
            )

            # Transform to pixel space: J * Sigma * J^T
            sx = float(self.pxl_per_cm_x)
            sy = float(self.pxl_per_cm_y)
            J = np.array([[sx, 0.0], [0.0, sy]], dtype=float)
            Sigma_px = J @ Sigma_xy @ J.T

            # Ensure symmetry
            Sigma_px = 0.5 * (Sigma_px + Sigma_px.T)

            # Eigenvalues and eigenvectors
            vals, vecs = np.linalg.eigh(Sigma_px)
            vals = np.maximum(vals, 0.0)  # Clamp

            # k-sigma factor (e.g., 2-sigma)
            k = 2.0
            idx_max = 1 if vals[1] >= vals[0] else 0
            idx_min = 1 - idx_max

            major_axis = k * math.sqrt(vals[idx_max])
            minor_axis = k * math.sqrt(vals[idx_min])

            # Center in pixels
            rx, ry, _ = robot_pose
            cx = int(rx * self.pxl_per_cm_x)
            cy = int((H - ry) * self.pxl_per_cm_y)

            # Angle in degrees
            vx = float(vecs[0, idx_max])
            vy = float(vecs[1, idx_max])
            angle = math.degrees(math.atan2(vy, vx))

            axes = (int(max(major_axis, 1.0)), int(max(minor_axis, 1.0)))

            ellipse_colour = (255, 255, 0)  # Yellow
            cv2.ellipse(frame, (cx, cy), axes, angle, 0.0, 360.0, ellipse_colour, 1, cv2.LINE_AA)
        except Exception:
            pass

    def _draw_robot(self, frame: np.ndarray, robot_pose: Pose, target: Optional[Point], H: float):
        """Draws the robot's position, orientation, and target line."""
        pose_colour = (70, 0, 160)
        rx, ry, rtheta = robot_pose
        cx = int(rx * self.pxl_per_cm_x)
        cy = int((H - ry) * self.pxl_per_cm_y)

        # Direction Vector
        dx = int(30 * math.cos(rtheta))
        dy = int(30 * -math.sin(rtheta))

        cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), pose_colour, 2, tipLength=0.6)
        cv2.circle(frame, (cx, cy), 5, pose_colour, -1)

        # Text Label
        pos_text = f"x={rx:.1f} y={ry:.1f}"
        cv2.putText(
            frame,
            pos_text,
            (cx + 12, cy - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            pose_colour,
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
            pose_colour,
            2,
            cv2.LINE_AA,
        )

        # --- Draw Target Line ---
        if target:
            tx, ty = target
            tx_px = int(tx * self.pxl_per_cm_x)
            ty_px = int((H - ty) * self.pxl_per_cm_y)
            cv2.line(frame, (cx, cy), (tx_px, ty_px), (0, 0, 0), 1)
            cv2.circle(frame, (tx_px, ty_px), 4, (0, 0, 0), -1)

    def _draw_trajectories(self, frame: np.ndarray, H: float):
        """Draws the estimated and predicted trajectories."""
        pose_colour = (70, 0, 160)
        if len(self.ekf_est_traj) > 1:
            for i in range(1, len(self.ekf_est_traj)):
                x1, y1 = self.ekf_est_traj[i - 1]
                x2, y2 = self.ekf_est_traj[i]
                pt1 = (int(x1 * self.pxl_per_cm_x), int((H - y1) * self.pxl_per_cm_y))
                pt2 = (int(x2 * self.pxl_per_cm_x), int((H - y2) * self.pxl_per_cm_y))
                cv2.line(frame, pt1, pt2, pose_colour, 2)

        if len(self.ekf_pred_traj) > 1:
            for i in range(1, len(self.ekf_pred_traj)):
                x1, y1 = self.ekf_pred_traj[i - 1]
                x2, y2 = self.ekf_pred_traj[i]
                pt1 = (int(x1 * self.pxl_per_cm_x), int((H - y1) * self.pxl_per_cm_y))
                pt2 = (int(x2 * self.pxl_per_cm_x), int((H - y2) * self.pxl_per_cm_y))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 1)

    def _draw_state_text(
        self,
        frame: np.ndarray,
        mission_state: MissionState,
        controller_state: RobotState,
    ):
        """Draws the current mission/controller state text."""
        state_text = controller_state.name if mission_state == MissionState.RUNNING else mission_state.name
        cv2.putText(
            frame, state_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2, cv2.LINE_AA
        )

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _detect_calibration_markers(self, img: np.ndarray) -> np.ndarray:
        """
        Detects the 4 green calibration markers in the `img` corners.
        :returns: sorted list of the 4 corner points: [TL, TR, BL, BR]
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Red heart target (reused for calibration markers in this setup?)
        # Note: The original code used red ranges for calibration markers.
        lo_red1 = np.array([0, 50, 50])
        hi_red1 = np.array([15, 255, 255])
        lo_red2 = np.array([165, 50, 50])
        hi_red2 = np.array([180, 255, 255])

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lo_red1, hi_red1),
            cv2.inRange(hsv, lo_red2, hi_red2),
        )

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 10:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    centroids.append([cX, cY])

        if len(centroids) != 4:
            raise RuntimeError(f"Calibration failed: Expected 4 markers, found {len(centroids)}")

        centroids = np.array(centroids, dtype=np.float32)

        # Sort points to match [TL, TR, BL, BR] order
        centroids = centroids[np.argsort(centroids[:, 1])]  # Sort by Y
        top = centroids[:2]
        bottom = centroids[2:]

        top = top[np.argsort(top[:, 0])]  # Sort Top by X
        bottom = bottom[np.argsort(bottom[:, 0])]  # Sort Bottom by X

        return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)

    def _detect_calibration_aruco(self, img, corners, ids) -> np.ndarray:
        """Helper to detect calibration markers using ArUco (unused in current flow but kept for reference)."""
        corners_centers = []
        for i, c in enumerate(corners):
            marker_id = ids[i][0]
            if marker_id != 1:
                pts = c[0]
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()
                corners_centers.append([marker_id, [cx, cy]])

        corners_centers.sort(key=lambda x: x[0])
        corners_centers = np.array([item[1] for item in corners_centers])

        if len(corners_centers) != 4:
            raise RuntimeError(f"Calibration failed: Expected 4 markers, found {len(corners_centers)}")
        return corners_centers

    def _merge_close_vertices(self, pts: np.ndarray, min_dist: float = 5.0) -> np.ndarray:
        """
        Merges vertices in a polygon that are closer than `min_dist` cm apart.
        """
        if len(pts) <= 2:
            return pts

        # Merge close points, averaging their positions
        new_pts = [pts[0]]
        for i in range(1, len(pts)):
            dist = np.linalg.norm(pts[i] - new_pts[-1])
            if dist > min_dist:
                new_pts.append(pts[i])
            else:
                new_pts[-1] = 0.5 * (new_pts[-1] + pts[i])

        if np.linalg.norm(np.array(new_pts[-1]) - np.array(new_pts[0])) < min_dist:
            new_pts.pop()

        return np.array(new_pts).reshape(-1, 2)

    def _is_visible(self, pt1: Point, pt2: Point, obstacles: List[Polygon]) -> bool:
        """
        Checks if the line segment between pt1 and pt2 is visible (not blocked by obstacles).
        """
        line = LineString([pt1, pt2])
        dist = line.length

        # If points are extremely close, consider them visible (probably the same point)
        if dist < 1e-6:
            return True

        # Trim the line slightly from both ends to avoid touching obstacle edges
        l_start = line.interpolate(0.001 * dist)
        l_end = line.interpolate(0.999 * dist)
        test_line = LineString([l_start, l_end])

        for poly in obstacles:
            if test_line.intersects(poly):
                return False
            if test_line.within(poly):
                return False

        return True

    def _construct_visibility_graph(
        self,
        obstacles: List[Polygon],
        start_pos: Optional[Point],
        goal_pos: Point,
    ) -> Tuple[nx.Graph, List[Point]]:
        """
        Constructs a visibility graph given polygon obstacles, and start & goal positions.
        """
        G = nx.Graph()
        obstacle_vertices = [Point(*pt) for poly in obstacles for pt in poly.exterior.coords[:-1]]
        all_pts = obstacle_vertices + [goal_pos]

        if start_pos:
            all_pts.append(start_pos)

        for i, pt in enumerate(all_pts):
            G.add_node(i, pos=pt)

        # Add obstacle edges
        for poly in obstacles:
            poly_pts = list(poly.exterior.coords)[:-1]
            num_pts = len(poly_pts)
            for i in range(num_pts):
                pt1 = Point(*poly_pts[i])
                pt2 = Point(*poly_pts[(i + 1) % num_pts])

                idx1 = self._find_point_index(pt1, all_pts)
                idx2 = self._find_point_index(pt2, all_pts)

                if idx1 is not None and idx2 is not None:
                    dist = math.dist(pt1, pt2)
                    G.add_edge(idx1, idx2, weight=dist)

        # Add visibility edges
        for (i, pt1), (j, pt2) in itertools.combinations(enumerate(all_pts), 2):
            if i == j:
                continue
            if G.has_edge(i, j):
                continue

            if self._is_visible(pt1, pt2, obstacles):
                dist = math.dist(pt1, pt2)
                G.add_edge(i, j, weight=dist)

        # Remove out-of-bounds nodes
        nodes_to_remove = []
        for n in G.nodes:
            p = G.nodes[n]["pos"]
            if p.x < 0 or p.x > self.map_width or p.y < 0 or p.y > self.map_height:
                nodes_to_remove.append(n)
        G.remove_nodes_from(nodes_to_remove)
        all_pts = [G.nodes[n]["pos"] for n in G.nodes]

        return G, all_pts

    def _find_point_index(self, pt: Point, all_pts: List[Point]) -> Optional[int]:
        """Finds the index of a Point in a list of Points, using approximate equality."""
        for i, pt2 in enumerate(all_pts):
            if math.isclose(pt.x, pt2.x, abs_tol=1e-4) and math.isclose(pt.y, pt2.y, abs_tol=1e-4):
                return i
        return None

    def init_from_img(self, img: np.ndarray):
        """
        Helper "constructor" for testing (init with RGB img instead of a camera).
        Assumes VisionSystem instance was created with __new__ (instead of __init__) and `self.map_width` and `self.map_height` were set.
        """
        self.img = img

        self.src_points = self._detect_calibration_markers(self.img)

        self.map_width_pxl = int(self.src_points[1][0] - self.src_points[0][0])
        self.map_height_pxl = int(self.src_points[2][1] - self.src_points[0][1])

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

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
