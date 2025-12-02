import math
from utils import Point, Pose, angle_between, normalize_angle
from enum import Enum


class RobotState(Enum):
	NAVIGATING = "NAVIGATING"
	AVOIDING = "AVOIDING"


class ThymioController:
	def __init__(self):
		self.KP_ROT = 120.0 	# P-gain for rotation
		self.MAX_SPEED = 120.0 	# Thymio units
		self.AVOID_SPEED = 80.0
		self.SENSOR_THRESHOLD = 2000 	# Threshold to trigger avoidance

		# Internal State
		self.state: RobotState = RobotState.NAVIGATING
		
        # Cooldown flag for an additional cycle in AVOIDING state
		self.just_finished_avoiding: bool = False
	
	def update(self, current_pose: Pose, target_pos: Point, sensor_data: list[int]) -> tuple[float, float]:
		"""
		Update control commands based on current pose, target position, and sensor data.
		"""

		# Check for obstacles in front 5 sensors
		max_front_sensor = max(sensor_data[0:5])
		is_obstacle_present = max_front_sensor > self.SENSOR_THRESHOLD
		
		
		if is_obstacle_present:
			self.state = RobotState.AVOIDING
			self.just_finished_avoiding = False
			
		elif self.state == RobotState.AVOIDING:
			
			if self.just_finished_avoiding:
				self.state = RobotState.NAVIGATING
				self.just_finished_avoiding = False # Reset flag
			else:
				self.just_finished_avoiding = True
				self.state = RobotState.AVOIDING # Stay in avoidance state for one more cycle
		
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
		turn_force = (sensor_data[0] + sensor_data[1] * 0.25) - (sensor_data[3] * 0.25 + sensor_data[4])

		# Base speed for avoidance
		base = self.AVOID_SPEED

		# Apply gain to turn force
		rotation = turn_force * 0.04

		l_speed = base + rotation
		r_speed = base - rotation

		return int(l_speed), int(r_speed)
