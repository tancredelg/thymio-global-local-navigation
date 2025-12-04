import time
import math
from utils import Point, Pose, RobotState, angle_between, normalize_angle


class ThymioController:
    def __init__(self):
        # Control parameters
        self.MAX_LIN_SPEED = 125.0
        """Maximum linear speed of the Thymio robot, in Thymio units."""
        self.KP_ROT = 120.0
        """Proportional gain for rotational control."""
        self.PROXH_SENSOR_THRESHOLD = 400
        """Minimum horizontal proximity sensor reading to consider an obstacle "detected"."""
        self.FRONT_SENSOR_MIN_DELTA = 30
        """Minimum difference between front sensor readings to consider an obstacle "detected"."""
        self.AVOID_LIN_SPEED = 125.0
        """Base speed during local obstacle avoidance."""
        self.AVOID_TURN_WEIGHT_MULTIPLIER = 0.025
        """Multiplier for sensor-derived turning weight during obstacle avoidance."""
        self.AVOID_DECAY_DURATION = 4.0
        """Time in seconds over which to "decay" from avoidance behavior back to navigation."""

        # State variables
        self.state: RobotState = RobotState.NAVIGATING
        self.last_avoid_time = 0.0

    def update(
        self,
        current_pose: Pose,
        target_point: Point,
        proxh_sensor_data: list[int],
    ) -> tuple[int, int]:
        """
        Updates control commands based on current state, target, and sensor data.

        :param current_pose: Current robot pose
        :param target_point: Target position to navigate to
        :param proxh_sensor_data: List of horizontal proximity sensor readings
        :return: Tuple of (left_wheel_speed, right_wheel_speed)
        """

        current_time = time.time()

        # Check for obstacles in front 5 sensors
        max_front_sensor = max(proxh_sensor_data[0:5])
        is_obstacle_present = max_front_sensor > self.PROXH_SENSOR_THRESHOLD

        if is_obstacle_present:  # Enter AVOIDING state (newly triggered)
            self.state = RobotState.AVOIDING
            self.last_avoid_time = current_time
            return self._avoid_obstacles(proxh_sensor_data)  # (active avoidance - use sensors directly)

        # Calculate navigation commands (base behavior)
        l_speed_nav, r_speed_nav = self._move_to_point(current_pose, target_point)

        match self.state:
            case RobotState.NAVIGATING:
                return l_speed_nav, r_speed_nav
            case RobotState.AVOIDING:
                # We are in the decay phase after initial avoidance

                # Calculate time since last avoidance trigger to determine decay progress
                time_since_last_avoid = current_time - self.last_avoid_time

                if time_since_last_avoid >= self.AVOID_DECAY_DURATION:
                    self.state = RobotState.NAVIGATING
                    return l_speed_nav, r_speed_nav

                # Still in decay phase - calculate normalized decay factor/progress to interpolate speeds
                # 1.0 = 100% Drive Straight, 0.0 = 100% Move to Point
                decay = max(0.0, 1.0 - (time_since_last_avoid / self.AVOID_DECAY_DURATION))

                l_speed_straight = self.MAX_LIN_SPEED
                r_speed_straight = self.MAX_LIN_SPEED

                # Linearly interpolate between straight and navigation speeds
                l_cmd = int(decay * l_speed_straight + (1.0 - decay) * l_speed_nav)
                r_cmd = int(decay * r_speed_straight + (1.0 - decay) * r_speed_nav)

                return l_cmd, r_cmd
            case _:
                raise ValueError(f"Unknown RobotState: {self.state}")

    def _move_to_point(self, current_pose: Pose, target_point: Point) -> tuple[int, int]:
        """
        Calculates wheel speeds to move towards a target point from a current pose,
        using a simple proportional controller for heading.
        """

        # Calculate heading error
        angle_to_target = angle_between(current_pose, target_point)
        heading_error = normalize_angle(angle_to_target - current_pose.theta)

        # As heading_error approaches 0, cos(error) approaches 1 -> full speed
        # As heading_error approaches 90 deg, cos(error) approaches 0 -> stop and turn
        linear_speed = self.MAX_LIN_SPEED * max(0.0, math.cos(heading_error))

        # Angular speed proportional to heading error
        angular_speed = self.KP_ROT * heading_error

        l_speed = int(linear_speed - angular_speed)
        r_speed = int(linear_speed + angular_speed)

        return l_speed, r_speed

    def _avoid_obstacles(self, proxh_sensor_data: list[int]) -> tuple[int, int]:
        """
        Calculates wheel speeds to avoid obstacles based on proximity sensor readings.
        """

        # Base avoidance weigths for front sensors
        # Weighs inner front sensors more heavily for turning
        outer_sensor_weight = 1.0  # Sensor 0 and 4
        inner_sensor_weight = 1.7  # Sensor 1 and 3

        # If both inner front sensors are very close in reading, we are likely facing a wall head-on
        # To avoid a deadlock, bias more strongly to turning (using outer sensors)
        if abs(proxh_sensor_data[1] - proxh_sensor_data[3]) < self.FRONT_SENSOR_MIN_DELTA:
            inner_sensor_weight = 0.0
            outer_sensor_weight = 2.3

        # Sum the weighted sensor readings to determine turn direction
        turn_weight = (
            proxh_sensor_data[0] * outer_sensor_weight
            + proxh_sensor_data[1] * inner_sensor_weight
            - proxh_sensor_data[3] * inner_sensor_weight
            - proxh_sensor_data[4] * outer_sensor_weight
        )

        angular_speed = turn_weight * self.AVOID_TURN_WEIGHT_MULTIPLIER
        l_speed = int(self.AVOID_LIN_SPEED + angular_speed)
        r_speed = int(self.AVOID_LIN_SPEED - angular_speed)

        return l_speed, r_speed
