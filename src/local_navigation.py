import numpy
import math

basic_speed = 100
obstacle_threshold = 200

avoid_gain = 1
avoid_scale = 8

kp_angle = 30 # Angle controller gain (for heading correction)

w_cam_pos = 0.9      
w_cam_theta = 0.95 

camera_available = True #if camra is blocked used odomoetry only

# State machine
state_goto = 0
state_avoid = 1
state_return = 2
state = state_goto

camera_x, camera_y = None
camera_theta = None
odom_x, odom_y = 0.0
odom_theta = 0.0

# Arbitrary goal to replace with target from path planner
goal_x = 100.0
goal_y = 0.0

def normalize_angle(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

# Weighted fusion (camera dominates odometry)
def fuse_pose():
    if camera_x is None or camera_theta is None:
        # camera not available → rely entirely on odometry (if they ask us)
        return odom_x, odom_y, odom_theta

    fused_x = w_cam_pos * camera_x + (1 - w_cam_pos) * odom_x
    fused_y = w_cam_pos * camera_y + (1 - w_cam_pos) * odom_y

    cx = math.cos(camera_theta)
    sx = math.sin(camera_theta)
    ox = math.cos(odom_theta)
    os = math.sin(odom_theta)

    fused_cos = w_cam_theta * cx + (1 - w_cam_theta) * ox
    fused_sin = w_cam_theta * sx + (1 - w_cam_theta) * os

    fused_theta = math.atan2(fused_sin, fused_cos) 
    return fused_x, fused_y, fused_theta 
@onevent
def prox():
    global state, motor_left_target, motor_right_target, camera_x, camera_y, camera_theta
    global odom_x, odom_y, odom_theta, goal_x, goal_y

    fused_x, fused_y, fused_theta = fuse_pose()

    dx = goal_x - fused_x
    dy = goal_y - fused_y
    theta_goal = math.atan2(dy, dx)

    heading_error = normalize_angle(theta_goal - fused_theta)

    left_side = prox_horizontal[0] + prox_horizontal[1] // 2
    right_side = prox_horizontal[4] + prox_horizontal[3] // 2

    obstacle = (left_side > obstacle_threshold or right_side > obstacle_threshold)

    #Sate machine
    if state == state_goto:
        if obstacle:
            state = state_avoid

    elif state == state_avoid:
        if not obstacle:
            state = state_return

    elif state == state_return:
        if obstacle:
            state = state_avoid
        else:
            # If almost facing the goal, return to navigation mode
            if abs(heading_error) < math.radians(6):
                state = state_goto

    #States
    if state == state_goto:
        correction = kp_angle * heading_error
        motor_left_target = basic_speed + correction
        motor_right_target = basic_speed - correction

    elif state == state_avoid:
        steer = (right_side - left_side) * avoid_gain // avoid_scale
        motor_left_target = basic_speed - steer
        motor_right_target = basic_speed + steer

    elif state == state_return:
        correction = kp_angle * heading_error
        motor_left_target = basic_speed + correction
        motor_right_target = basic_speed - correction

    # Clip outputs
    max_speed = 300
    motor_left_target = max(-max_speed, min(max_speed, int(motor_left_target)))
    motor_right_target = max(-max_speed, min(max_speed, int(motor_right_target)))
