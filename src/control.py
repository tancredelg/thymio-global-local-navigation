import math
from utils import Point, Pose, angle_between, normalize_angle
from enum import Enum


# FSM state types
class RobotState(Enum):
    NAVIGATING = "NAVIGATING"
    AVOIDING = "AVOIDING"


class ThymioController:
    def __init__(self):
        # Tuning Constants
        self.KP_ROT = 40.0  # P-gain for rotation
        self.MAX_SPEED = 100.0  # Thymio units
        self.AVOID_SPEED = 60.0
        self.SENSOR_THRESHOLD = 2000  # Threshold to trigger avoidance

        # Internal State
        self.state: RobotState = RobotState.NAVIGATING

    def update(self, current_pose: Pose, target_pos: Point, sensor_data: list[int]) -> tuple[float, float]:
        """
        Update control commands based on current pose, target position, and sensor data.

        :param current_pose: Current estimated pose of the robot (typcally from a state estimator - e.g., EKF)
        :param target_pos: Target _**position**_ to navigate to (next waypoint in path given by path planner)
        :param sensor_data: Additional sensor data (e.g., proximity sensors) for obstacle avoidance
        """

        # Check for obstacles in front 5 sensors
        max_front_sensor = max(sensor_data[0:5])
        if max_front_sensor > self.SENSOR_THRESHOLD:
            self.state = RobotState.AVOIDING
        else:
            self.state = RobotState.NAVIGATING

        match self.state:
            case RobotState.AVOIDING:
                return self._avoid_obstacles(sensor_data)
            case RobotState.NAVIGATING:
                return self._move_to_point(current_pose, target_pos)

    def _move_to_point(self, current_pose: Pose, target_pos: Point) -> tuple[float, float]:
        """
        P-Controlled navigation to a target point.
        Uses a proportional controller to compute linear and angular speeds to move towards the target position.

        :param current_pose: Current estimated pose of the robot
        :param target_pos: Target _**position**_ to navigate to
        :return: (left_wheel_speed, right_wheel_speed)
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

        :param sensor_data: List of proximity sensor readings
        :return: (left_wheel_speed, right_wheel_speed)
        """

        # Simple Braitenberg-style logic:
        # "If left sensor sees something, speed up right wheel (turn right)"
        # "If right sensor sees something, speed up left wheel (turn left)"

        # Map sensors to weights (simplified example)
        # sensors: [LeftMost, Left, Center, Right, RightMost]

        # Calculate a "turning force" based on sensor difference
        # Positive = Turn Right, Negative = Turn Left
        turn_force = (sensor_data[0] + sensor_data[1] // 2) - (sensor_data[3] // 2 + sensor_data[4])

        # Base speed for avoidance
        base = self.AVOID_SPEED

        # Apply gain to turn force
        rotation = turn_force * 0.05

        l_speed = base + rotation
        r_speed = base - rotation

        return int(l_speed), int(r_speed)
