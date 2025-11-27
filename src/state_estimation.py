import numpy as np
from utils import Pose


class ExtendedKalmanFilter:
    """
    An extended Kalman filter (EKF) for nonlinear state estimation.
    """

    def __init__(self, initial_pose: Pose, initial_covariance: np.ndarray):
        """
        Initializes the EKF with an initial pose and covariance.

        :param initial_pose: Initial pose estimate `Pose(x, y, theta)`
        :param initial_covariance: Initial state covariance matrix (np.ndarray of shape (3, 3))
        """

        assert initial_covariance.shape == (3, 3), "Initial covariance must be a 3x3 matrix."

        self.state: Pose = initial_pose
        """Current state (pose) estimate"""
        self.covariance = initial_covariance
        """Current state covariance matrix"""

        # TODO: Complete initialization
        raise NotImplementedError("Implementation not complete.")

    def get_state(self) -> Pose:
        return self.state

    def predict(self, speed_l: float, speed_r: float, dt: float) -> None:
        raise NotImplementedError

    def update(self, measured_pose: Pose) -> None:
        raise NotImplementedError
