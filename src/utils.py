from typing import NamedTuple
import math
from collections.abc import Callable


# Define a simple Point class for 2D coordinates - simply for improved type clarity & safety
# (Behaves just like a tuple but with named fields, and constrained to two floats)
class Point(NamedTuple):
    x: float
    y: float


# And similarly for 2D Pose (x, y, theta)
class Pose(NamedTuple):
    x: float
    y: float
    theta: float  # Orientation in radians


# Type alias for heuristic functions
type HeuristicFunction = Callable[[Point, Point], float]


def euclidean_distance(p1: Point, p2: Point) -> float:
    return math.dist(p1, p2)  # (math.dist == Euclidean distance)


def angle_between(p1: Point, p2: Point) -> float:
    """Calculate the angle (in radians) of the vector from p1 to p2."""
    return math.atan2(p2.y - p1.y, p2.x - p1.x)
