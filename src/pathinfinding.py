import heapq
import networkx as nx
from utils import Point, HeuristicFunction, euclidean_distance


def reconstruct_path(came_from: dict[int, int], current: int) -> list[int]:
    """
    Backtracks from a goal node (ID) to start node (ID) to generate the final path list ordered from start to goal.

    :param came_from: Dict mapping node ID to its parent node ID
    :param current: Current node ID (goal)
    :return: List of node IDs from start to goal
    """

    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    return total_path[::-1]  # Reverse to get Start -> Goal


def find_path(
    graph: nx.Graph,
    start_node: int,
    goal_node: int,
    heuristic: HeuristicFunction = euclidean_distance,
) -> list[int]:
    """
    Finds the shortest path in a NetworkX graph from a start to goal node using the A* algorithm.

    :param graph: `networkx.Graph` where nodes have attribute `pos: Point` and edges have attribute `weight: float`
    :param start_node: Starting node ID
    :param goal_node: Goal node ID
    :param heuristic: Function to estimate cost between two nodes, with signature `(Point, Point) -> float` (default: Euclidean distance)
    :return: List of node IDs representing the path from start to goal.
    """

    # 1. Initialize Open Set (Priority Queue)
    # Stores tuples of (f_cost, node_id).
    # Heapq is a min-heap, so always pops the item with the lowest first element (lowest f_cost).
    open_set: list[tuple[float, int]] = []
    heapq.heappush(open_set, (0, start_node))

    # 2. Track "Parent" pointers for path reconstruction
    came_from: dict[int, int] = {}

    # 3. Track g_cost (Cost from start to current node)
    # Default to infinity for all nodes except start
    g_cost: dict[int, float] = {node: float("inf") for node in graph.nodes}
    g_cost[start_node] = 0

    # 4. Track f_cost (Estimated total cost: g + h)
    # We don't strictly need to store this for all nodes, but it helps debugging
    f_cost: dict[int, float] = {node: float("inf") for node in graph.nodes}
    f_cost[start_node] = euclidean_distance(graph.nodes[start_node]["pos"], graph.nodes[goal_node]["pos"])

    # Keep track of items in the heap to allow for "Lazy Deletion"
    # (Python's heapq doesn't support updating priorities, so we just add duplicates
    # and ignore the old, worse ones when we pop them).
    open_set_hash: set[int] = {start_node}

    while open_set:
        # Get node with lowest f_cost
        current_f, current = heapq.heappop(open_set)

        # If we reached the goal, we are done!
        if current == goal_node:
            # Return the reconstructed path (need to backtrack to start, and reverse it)
            return reconstruct_path(came_from, current)

        open_set_hash.discard(current)

        # Explore neighbors
        # graph[current] gives a dict of neighbors: {neighbor_id: {'weight': dist}}
        for neighbor in graph.neighbors(current):
            # Calculate tentative g_cost
            edge_weight = graph[current][neighbor]["weight"]
            tentative_g_cost = g_cost[current] + edge_weight

            # If this path to neighbor is better than any previous one
            if tentative_g_cost < g_cost[neighbor]:
                # Record this path
                came_from[neighbor] = current
                g_cost[neighbor] = tentative_g_cost

                # Calculate f_cost = g_cost + heuristic
                h = heuristic(graph.nodes[neighbor]["pos"], graph.nodes[goal_node]["pos"])
                f = tentative_g_cost + h
                f_cost[neighbor] = f

                # Add to priority queue if not already there
                # (Or if it is there, we add the duplicate with better score.
                # Ideally we check, but for small graphs, adding duplicates is fine/faster).
                heapq.heappush(open_set, (f, neighbor))
                open_set_hash.add(neighbor)

    # If we get here, open_set is empty but goal was not reached
    print("Error: No path found!")
    return []
