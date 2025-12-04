# main.py
import asyncio
import time
import numpy as np
import argparse
import traceback
from typing import Tuple, Optional, List

from tdmclient import ClientAsync, Node

# Project Modules
from vision import VisionSystem
from control import ThymioController
from pathfinding import find_path
from utils import Pose, Point, euclidean_distance, MissionState
from extended_kalman_filter import EKF
from thymio_math_model import Thymio

# --- Configuration Constants ---
INITIAL_POSE = Pose(0, 0, 0)
INITIAL_COVARIANCE = np.eye(3) * 1e-1
# Camera covariance (tuned)
CAMERA_COVARIANCE = np.diag(
    [
        0.0011857353432198614,
        0.0017873989613467563,
        6.1009773737464586e-05,
    ]
)

WAYPOINT_THRESHOLD = 1.5  # cm
DT_NOMINAL = 0.1  # seconds (10Hz)
KIDNAP_THRESHOLD = 200  # Threshold for ground sensor delta (lower = lifted)
RESTART_DELAY = 2.5  # Seconds to wait after being put down


# =============================================================================
# Helper Functions
# =============================================================================


def perform_planning(vis: VisionSystem, ekf: EKF) -> Tuple[Optional[List[Point]], int]:
    """
    Locates the robot, builds the visibility graph, and plans a path to the goal.
    Also resets the EKF state to the newly found pose.

    :param vis: VisionSystem instance
    :param ekf: EKF instance
    :return: Tuple (list of waypoints, start_node_index) or (None, -1) if failed
    """
    print("[Planning] Locating Robot...")
    start_pose = None

    # Retry a few times if needed
    for _ in range(3):
        start_pose = vis.get_robot_pose()
        if start_pose:
            break
        time.sleep(0.5)

    if start_pose is None:
        print("[Planning] Could not locate robot.")
        return None, -1

    # Augment graph with robot position
    graph, start_node_idx, goal_node_idx = vis.add_robot_to_graph(start_pose)

    # Find path using A*
    waypoints_idx = find_path(graph, start_node_idx, goal_node_idx)
    waypoints = [graph.nodes[i]["pos"] for i in waypoints_idx]

    print(f"[Planning] Path found with {len(waypoints)} waypoints.")

    # Reset EKF to visual pose
    ekf.x = np.array([[start_pose.x], [start_pose.y], [start_pose.theta]], dtype=float)
    ekf.P = INITIAL_COVARIANCE.copy()

    # Initialize visualization
    vis.init_visu(graph, start_node_idx, goal_node_idx, waypoints)
    vis.ekf_est_traj.append((float(start_pose.x), float(start_pose.y)))

    return waypoints, start_node_idx


async def initialize_thymio() -> Tuple[ClientAsync, Node]:
    """Connects to the Thymio and locks the node."""
    print("[Main] Connecting to Thymio via ClientAsync...")
    client = ClientAsync()
    node = await client.wait_for_node()
    await node.lock()
    print("[Main] Thymio Connected and Locked.")

    await node.wait_for_variables(
        [
            "prox.horizontal",
            "prox.ground.delta",
            "motor.left.target",
            "motor.right.target",
            "motor.left.speed",
            "motor.right.speed",
        ]
    )
    return client, node


async def cleanup(
    client: ClientAsync, node: Node, vis: VisionSystem, mission_state: MissionState, controller_state
):
    """Stops motors, unlocks robot, and handles final visualization cleanup."""
    print("[Main] Cleanup initiated...")

    # Stop Motors
    try:
        await node.set_variables({"motor.left.target": [0], "motor.right.target": [0]})
        node.flush()
    except Exception:
        pass

    # Unlock Robot
    try:
        await node.unlock()
        print("[Main] Robot unlocked.")
    except Exception:
        pass

    # Close Client
    try:
        await client.close()
        print("[Main] Client disconnected.")
    except Exception:
        pass

    # Wait for user to close visualization
    print("[Main] Press 'q' in the visualization window to exit.")
    while True:
        # We use get_robot_pose() just to keep the camera feed alive if needed
        if vis.update_robot_visu(mission_state, controller_state, vis.get_robot_pose()):
            break

    vis.release()
    print("[Main] Cleanup complete. Exiting.")


# =============================================================================
# Main Loop
# =============================================================================


async def run_robot(camera_index: int, warmup_time: int):
    """
    Main asynchronous control loop.
    """
    # --- 1. Initialize Systems ---
    print(f"[Main] Initializing Vision System on camera {camera_index}...")
    vis = VisionSystem(camera_index=camera_index, warmup_time=warmup_time)
    controller = ThymioController()

    thymio_model = Thymio()
    thymio_model.dt = DT_NOMINAL
    thymio_model.freq = 1.0 / DT_NOMINAL

    ekf = EKF(
        thymio_model,
        np.array([INITIAL_POSE.x, INITIAL_POSE.y, INITIAL_POSE.theta]),
        INITIAL_COVARIANCE,
        CAMERA_COVARIANCE,
    )

    # --- 2. Map Construction & Initial Planning ---
    print("[Main] Constructing Map...")
    try:
        vis.build_static_map(cspace_padding=3.0)
        waypoints, _ = perform_planning(vis, ekf)

        if waypoints is None:
            raise RuntimeError("Initial planning failed.")

    except Exception as e:
        print(f"[Main] Mapping/Planning failed: {e}")
        vis.release()
        return

    # --- 3. Connect to Thymio ---
    client, node = await initialize_thymio()

    # --- 4. Control Loop State ---
    current_waypoint_idx = 0
    l_cmd, r_cmd = 0, 0
    thymio_model.v_l, thymio_model.v_r = 0, 0

    last_time = time.time()
    mission_state = MissionState.RUNNING
    restart_start_time = 0.0

    print("[Main] Starting Control Loop. Press Ctrl+C to stop.")

    try:
        while True:
            start_loop_time = time.time()

            # --- A. Time Management ---
            now = time.time()
            dt_real = now - last_time
            last_time = now

            # Clamp dt to avoid explosions if lag occurs
            dt_real = max(0.0, min(dt_real, 0.5))
            if dt_real == 0:
                dt_real = DT_NOMINAL

            # Update model physics
            thymio_model.dt = dt_real
            thymio_model.freq = 1.0 / dt_real

            # --- B. Sensing ---
            # 1. Vision
            vision_pose_measurement = vis.get_robot_pose()
            estimated_pose = None
            ekf_pred_pose = None

            # 2. Odometry (Motor speeds)
            try:
                left_speed_meas = node.v.motor.left.speed
                right_speed_meas = node.v.motor.right.speed
            except Exception:
                left_speed_meas, right_speed_meas = 0, 0

            thymio_model.v_l = thymio_model.thymio_unit_to_speed(left_speed_meas)
            thymio_model.v_r = thymio_model.thymio_unit_to_speed(right_speed_meas)

            # 3. Proximity
            prox_sensors = list(node.v.prox.horizontal)
            ground_sensors = list(node.v.prox.ground.delta)

            # --- C. Mission State Logic (Kidnapping) ---
            is_lifted = max(ground_sensors) < KIDNAP_THRESHOLD

            if mission_state == MissionState.RUNNING:
                if is_lifted:
                    print("[Mission] KIDNAPPED! Stopping motors.")
                    mission_state = MissionState.KIDNAPPED

            elif mission_state == MissionState.KIDNAPPED:
                if not is_lifted:
                    print("[Mission] Robot grounded. Stabilizing...")
                    mission_state = MissionState.RESTARTING
                    restart_start_time = now

            elif mission_state == MissionState.RESTARTING:
                if is_lifted:
                    print("[Mission] Kidnapped again!")
                    mission_state = MissionState.KIDNAPPED
                elif (now - restart_start_time) > RESTART_DELAY:
                    print("[Mission] Stabilized. Re-planning...")
                    try:
                        new_waypoints, _ = perform_planning(vis, ekf)
                        if new_waypoints:
                            waypoints = new_waypoints
                            current_waypoint_idx = 0
                            mission_state = MissionState.RUNNING
                            print(f"[Mission] Resumed with {len(waypoints)} waypoints.")
                        else:
                            print("[Mission] Could not locate robot. Retrying...")
                    except Exception as e:
                        print(f"[Mission] Re-planning failed: {e}")

            # --- D. State Estimation & Control (Only if RUNNING) ---
            target = None

            if mission_state == MissionState.RUNNING:
                # 1. Predict
                ekf.predict_step()
                pred_vec = ekf.x.flatten()
                ekf_pred_pose = Pose(pred_vec[0], pred_vec[1], pred_vec[2])

                # 2. Update (if camera saw tag)
                if vision_pose_measurement is not None:
                    z = np.array(
                        [
                            vision_pose_measurement.x,
                            vision_pose_measurement.y,
                            vision_pose_measurement.theta,
                        ]
                    )
                    ekf.update_step(z)

                # 3. Get final estimate
                est_vec = ekf.x.flatten()
                estimated_pose = Pose(est_vec[0], est_vec[1], est_vec[2])

                # 4. Path Following
                if current_waypoint_idx < len(waypoints):
                    target = waypoints[current_waypoint_idx]
                    dist = euclidean_distance(Point(estimated_pose.x, estimated_pose.y), target)

                    if dist < WAYPOINT_THRESHOLD:
                        print(f"[Nav] Reached Waypoint {current_waypoint_idx}")
                        current_waypoint_idx += 1

                    if current_waypoint_idx >= len(waypoints):
                        print("[Nav] GOAL REACHED! Stopping.")
                        l_cmd, r_cmd = 0, 0
                    else:
                        target = waypoints[current_waypoint_idx]
                        l_cmd, r_cmd = controller.update(estimated_pose, target, prox_sensors)
            else:
                # If not RUNNING, stop motors
                l_cmd, r_cmd = 0, 0

            # --- E. Actuation ---
            v_cmd = {
                "motor.left.target": [int(l_cmd)],
                "motor.right.target": [int(r_cmd)],
            }
            await node.set_variables(v_cmd)
            node.flush()  # Force send immediately

            # --- F. Visualization ---
            # Returns True if 'q' is pressed
            if vis.update_robot_visu(
                mission_state,
                controller.state,
                estimated_pose,
                target,
                ekf_pred_pose=ekf_pred_pose,
                ekf_cov=ekf.P if estimated_pose is not None else None,
            ):
                print("[Main] User requested exit.")
                break

            # --- G. Sleep to maintain rate ---
            elapsed = time.time() - start_loop_time
            if elapsed < DT_NOMINAL:
                await client.sleep(DT_NOMINAL - elapsed)

    except asyncio.CancelledError:
        print("[Main] Loop cancelled.")
    except Exception as e:
        print(f"[Main] Unexpected error: {e}")
        traceback.print_exc()
    finally:
        await cleanup(client, node, vis, mission_state, controller.state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thymio Navigation System")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--warmup", type=int, default=5, help="Camera warmup time in seconds (default: 5)")

    args = parser.parse_args()

    try:
        asyncio.run(run_robot(args.camera, args.warmup))
    except KeyboardInterrupt:
        # This catches Ctrl+C in the terminal
        print("\n[System] Interrupted by user.")
