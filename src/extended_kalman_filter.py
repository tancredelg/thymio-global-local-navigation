import numpy as np
import math


class EKF:
    def __init__(self, Thymio, x0, p0, R):
        self.model = Thymio
        self.x = np.array(x0, dtype=float).reshape(3, 1)
        self.P = np.array(p0, dtype=float).reshape(3, 3)
        self.R = np.array(R, dtype=float).reshape(3, 3)
        self.POS_JUMP_THRESHOLD_CM = 30.0  # Threshold for kidnap detection

    def predict_step(self):
        # print("[EKF] ENTER predict_step, x =", self.x.ravel())

        dF = self.model.jacobian_dF(self.x)
        Q = self.model.compute_Q(self.x)

        x_n_plus_one = self.model.state_extrapolation_f(self.x)
        P_n_plus_one = dF @ self.P @ dF.T + Q

        self.x = x_n_plus_one
        self.P = P_n_plus_one
        # print("[EKF] EXIT predict_step, x =", self.x.ravel())
        return True

        # print(f"EKF predict: {x_n_plus_one[0], x_n_plus_one[1], x_n_plus_one[2]}")

    def update_step(self, z) -> bool:
        # print("[EKF] ENTER update_step, z =", z)

        H = np.eye(3)
        z = np.array(z, dtype=float).reshape(3, 1)

        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)

        x_n_current = self.x + K @ (z - H @ self.x)

        P_n_current = (np.eye(3) - K @ H) @ self.P @ (np.eye(3) - K @ H).T + K @ self.R @ K.T

        self.x = x_n_current
        self.P = P_n_current

        # print("[EKF] EXIT update_step, x =", self.x.ravel())

        return True
