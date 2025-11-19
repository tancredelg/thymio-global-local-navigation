# Global-Local Robot Navigation Through Obstacle-Filled Environments

The final project for EPFL's [**MICRO-452: Basics of Mobile Robotics**](https://edu.epfl.ch/coursebook/en/basics-of-mobile-robotics-MICRO-452) course. The project is intended to address and put into practice (in a basic flavour) key aspects of mobile robotics covered along the course: vision, mapping, state estimation, global and local navigation, and motion control.

![Example environment](example_environment.png)

## 1. Physical Setup

Our project tasks a Thymio II robot - aided by a downward-facing global camera - with navigating through an obstacle-filled environment to a marked goal location.

### 1.1 Environment

The environment features permanent/static obstacles as well as ephemeral ones, requiring a non-trivial navigation strategy working on both a global and local scale:

-   The **static obstacles** part of the permanent environment may vary in size, but are always <u>polygonal in shape</u> and <u>easily distinguishable from the ground (and robot) by colour</u>.
-   The **ephemeral obstacles** however, are dynamically placed mid-run, in ways that disrupt the robot's navigation and create a need for local obstacle avoidance. They have <u>no guaranteed shape or colour</u>, but are always <u>around the size of the robot or smaller</u>, and will only be placed <u>in locations that force local avoidance without changing the optimal path to the goal</u> (e.g. not in the middle of narrow "choke points", nor at the apex of sharp turns). _They are expected to be detected by the Thymio's onboard proximity sensors, but not by the global camera._

### 1.2 Robot

The **Thymio II** robot is a small differential-drive mobile robot equipped with a variety of onboard sensors, including proximity sensors along the curved front, the back, and underneath. It is programmed to follow navigation commands received from, and send sensor data to, an external computer - either wirelessly or by USB. In our project, we interface with the the robot via its Python API.

### 1.3 Camera

The global camera, mounted above the environment and looking straight down, is a standard webcam capable of streaming <u>1920x1080, RGB video at 30 FPS</u>. It is connected to a computer and used for global localization of the robot, as well as mapping the static environment + planning paths on it.

## 2. Software Setup

All written in python3.

Key parts of the implementation:

1. **Mapping:** construction of a visibility graph of the static environment via feature detection on camera images
2. **Path planning:** global path planning via A\* on the visibility graph
3. **Localization:** global localization of the robot via camera tracking of visual markers on the robot body
4. **State estimation:** EKF fusing camera-based global localization and Thymio odometry
5. **Local navigation:** following waypoints of the global path while avoiding ephemeral obstacles detected by Thymio proximity sensors
