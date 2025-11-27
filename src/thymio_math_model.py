import numpy as np
import matplotlib.pyplot as plt 
import math 


class Thymio: 
    def __init__(self): 
        self.L = 9.5        # axle length (cm)
        self.R = 2.2        # wheel radius (cm)
        self.freq = 100     # Hz
        self.v_l = 0.0      # cm/s
        self.v_r = 0.0      # cm/s

        self.dt = 1.0 / self.freq

        # Covariances of the model (Q_model) 
        self.sigma_x_model = 0.01        #std in x (cm)
        self.sigma_y_model = 0.01        #std in y (cm)
        self.sigma_theta_model = 0.001   #std in theta (rad)

        # Input Wheel Speed Covariances 
        self.sigma_vl = 0.01              # std in v_l 
        self.sigma_vr = 0.01              # std in v_r 


    def wrap_angle(self, angle): 
        return math.atan2(math.sin(angle), math.cos(angle))

    # Thymio provides speed in thymio units which is 1 Thymio unit≈0.43478 mm/s
    def thymio_unit_to_speed(self, thymio_units): 
        return (thymio_units * 0.43478) / 10.0 
     
    def wheel_speed_map(self): 
        V = (1.0 / 2.0) * (self.v_l + self.v_r)
        w = (1.0 / self.L) * (self.v_r - self.v_l)
        return V, w

    def state_extrapolation_f(self, x):
        V, w = self.wheel_speed_map()
        x, y, theta = float(x[0]), float(x[1]), float(x[2])
        return np.array([[x + V*math.cos(theta)*self.dt],
                        [y + V*math.sin(theta)*self.dt],
                        [theta + w*self.dt]])
    
    def jacobian_dF(self, x): 
        theta = float(x[2])
        V, _ = self.wheel_speed_map() 

        F = np.eye(3) 
        F[0, 2] = -V * self.dt * math.sin(theta) 
        F[1, 2] = V * self.dt * math.cos(theta) 

        return F 
    
    # Jacobian to map the input noise into the model prediction 
    def jacobian_G(self, x): 
        theta = float(x[2])

        G = np.array([[ 0.5 * self.dt * math.cos(theta),   0.5 * self.dt * math.cos(theta) ], 
                      [ 0.5 * self.dt * math.sin(theta),   0.5 * self.dt * math.sin(theta)],
                      [ -self.dt / self.L,               self.dt / self.L]])
        return G 
    
    
    ### hard note: the wheel speed covariance must be either selected "randomly"
    ### or through experimentation. 
    def compute_Q(self, x):
        G = self.jacobian_G(x)

        sigma_u = np.diag([self.sigma_vl**2, self.sigma_vr**2])

        Q_process = np.diag([self.sigma_x_model**2,
                             self.sigma_y_model**2,
                             self.sigma_theta_model**2])

        return G @ sigma_u @ G.T + Q_process






    
