import math
from utils import Point, Pose, angle_between, normalize_angle
from enum import Enum
import time


class RobotState(Enum):
    NAVIGATING = "NAVIGATING"
    AVOIDING = "AVOIDING"


class ThymioController:
    def __init__(self):
        self.KP_ROT = 120.0  # P-gain for rotation
        self.MAX_SPEED = 125.0  # Thymio units
        self.AVOID_SPEED = 125.0
        self.SENSOR_THRESHOLD = 400  # Threshold to trigger avoidance
        self.AVOIDANCE_DURATION = 4.0  # Number of seconds to stay in avoidance state
        self.last_avoidance_time = 0.0
        self.FRONT_SENSOR_MIN_DELTA = 30

        # Internal State
        self.state: RobotState = RobotState.NAVIGATING

    def update(self, current_pose: Pose, target_pos: Point, sensor_data: list[int]) -> tuple[float, float]:
        """
        Update control commands based on current pose, target position, and sensor data.
        """
        current_time = time.time()

        # Check for obstacles in front 5 sensors
        max_front_sensor = max(sensor_data[0:5])
        is_obstacle_present = max_front_sensor > self.SENSOR_THRESHOLD

        # Calculate Navigation Component (Base behavior)
        nav_l, nav_r = self._move_to_point(current_pose, target_pos)

        if is_obstacle_present:
            self.state = RobotState.AVOIDING
            self.last_avoidance_time = current_time
            # Active avoidance: use sensors directly
            return self._avoid_obstacles(sensor_data)

        elif self.state == RobotState.AVOIDING:
            elapsed = current_time - self.last_avoidance_time
            if elapsed < self.AVOIDANCE_DURATION:
                # Decay phase: Blend "Drive Straight" with "Move to Point"
                # 1.0 = Drive Straight, 0.0 = Move to Point
                decay = max(0.0, 1.0 - (elapsed / self.AVOIDANCE_DURATION))

                # Drive straight component
                straight_l = self.MAX_SPEED
                straight_r = self.MAX_SPEED

                # Blend controls
                l_cmd = decay * straight_l + (1.0 - decay) * nav_l
                r_cmd = decay * straight_r + (1.0 - decay) * nav_r

                return int(l_cmd), int(r_cmd)
            else:
                self.state = RobotState.NAVIGATING

        # Default: Navigation
        return nav_l, nav_r

    def _move_to_point(self, current_pose: Pose, target_pos: Point) -> tuple[float, float]:
        """
        P-Controlled navigation to a target point.
        """

        # Calculate heading error
        angle_to_target = angle_between(current_pose, target_pos)
        heading_error = normalize_angle(angle_to_target - current_pose.theta)

        # Calculate (P-controlled) angular and (regulated) linear speeds
        angular_speed = self.KP_ROT * heading_error

        # As heading_error approaches 0, cos(error) approaches 1 -> full speed
        # As heading_error approaches 90 deg, cos(error) approaches 0 -> stop and turn
        linear_speed = self.MAX_SPEED * max(0.0, math.cos(heading_error))

        print(
            f"Heading error: {heading_error:.2f} rad, Linear speed: {linear_speed:.2f}, Angular speed: {angular_speed:.2f}"
        )

        l_speed = linear_speed - angular_speed
        r_speed = linear_speed + angular_speed

        return int(l_speed), int(r_speed)

    def _avoid_obstacles(self, sensor_data: list[int]) -> tuple[float, float]:
        """
        Simple obstacle avoidance behavior based on proximity sensor readings.
        """

        # Simple Braitenberg-style logic:
        # "If left sensor sees something, speed up right wheel (turn right)"
        # "If right sensor sees something, speed up left wheel (turn left)"

        # Map sensors to weights (simplified example)
        # sensors: [LeftMost, Left, Center, Right, RightMost]

        # Calculate a "turning force" based on sensor difference
        # Positive = Turn Right, Negative = Turn Left
        inner_sensor_weight = 1.7
        outer_sensor_weight = 1.0
        if abs(sensor_data[1] - sensor_data[3]) < self.FRONT_SENSOR_MIN_DELTA:
            inner_sensor_weight = 0.0
            outer_sensor_weight = 2.3

        turn_weight = (
            sensor_data[0] * outer_sensor_weight
            + sensor_data[1] * inner_sensor_weight
            - sensor_data[3] * inner_sensor_weight
            - sensor_data[4] * outer_sensor_weight
        )

        # Apply gain to turn force
        angular_speed = turn_weight * 0.025

        l_speed = self.AVOID_SPEED + angular_speed
        r_speed = self.AVOID_SPEED - angular_speed

        return int(l_speed), int(r_speed)
