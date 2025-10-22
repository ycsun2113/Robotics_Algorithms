import numpy as np
from utils import get_collision_fn_PR2, load_env, execute_trajectory, draw_sphere_marker
from pybullet_tools.utils import connect, disconnect, get_joint_positions, wait_if_gui, set_joint_positions, joint_from_name, get_link_pose, link_from_name
from pybullet_tools.pr2_utils import PR2_GROUPS
import time
### YOUR IMPORTS HERE ###
from queue import PriorityQueue
from utils import draw_line
#defines a basic node class
#defines a basic node class
class Node:
    def __init__(self, x_in, y_in, theta_in, g_in):
        self.x = x_in
        self.y = y_in
        self.theta = theta_in
        self.cost = g_in

    def printme(self):
        print("\tNode :", "x =", self.x, "y =",self.y, "theta =", self.theta)

    def xytg(self):
        return self.x, self.y, self.theta, self.cost

    def __eq__(self, other):
        return (self.x, self.y, self.theta)==(other.x, other.y, other.theta)
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))
    

def action_cost(nx, ny, ntheta, mx, my, mtheta):
    cost = np.sqrt((nx - mx)**2 + (ny - my)**2 + min(abs(ntheta - mtheta), 2*np.pi - abs(ntheta - mtheta))**2)
    return cost

def heuristic(nx, ny, ntheta, gx, gy, gtheta):
    h = np.sqrt((nx - gx)**2 + (ny - gy)**2 + min(abs(ntheta - gtheta), 2*np.pi - abs(ntheta - gtheta))**2)
    return h

def total_cost(prev_g, prev_x, prev_y, prev_theta, x, y, theta, gx, gy, gtheta):
    g_cost = prev_g + action_cost(prev_x, prev_y, prev_theta, x, y, theta)
    h = heuristic(x, y, theta, gx, gy, gtheta)
    return (0.35*g_cost + 0.65*h)

#########################

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2doorway.json')

    # define active DoFs
    base_joints = [joint_from_name(robots['pr2'], name) for name in PR2_GROUPS['base']]

    collision_fn = get_collision_fn_PR2(robots['pr2'], base_joints, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, -1.3, -np.pi/2)))

    # Example use of setting body poses
    # set_pose(obstacles['ikeatable6'], ((0, 0, 0), (1, 0, 0, 0)))

    # Example of draw 
    # draw_sphere_marker((0, 0, 1), 0.1, (1, 0, 0, 1))
    
    start_config = tuple(get_joint_positions(robots['pr2'], base_joints))
    goal_config = (2.6, -1.3, -np.pi/2)
    path = []
    start_time = time.time()
    ### YOUR CODE HERE ###

    sphere_radius = 0.03

    # Set the start config and draw the start point in the world:
    start_x, start_y, start_theta = start_config
    start_cost = 0
    start_position = (start_x, start_y, 0.2)
    draw_sphere_marker(start_position, sphere_radius, (1, 0, 0, 1))

    # Set the goal config and draw the goal point in the world:
    goal_x, goal_y, goal_theta = goal_config
    goal_position = (goal_x, goal_y, 0.2)
    draw_sphere_marker(goal_position, sphere_radius, (0, 1, 0, 1))

    neighbors_dir_4 = [(1,0), (0,1), (-1,0), (0,-1)]
    neighbors_dir_8 = [(1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1)]
    angles = [0, np.pi/2, np.pi, -np.pi/2]

    directions = neighbors_dir_4
    # directions = neighbors_dir_8

    step_size = 0.1
    threshold = 0.2
    iter = 0
    success = False

    q = PriorityQueue()
    queue_id = 0
    start_node = Node(start_x, start_y, start_theta, start_cost)
    q.put((total_cost(0, start_x, start_y, start_theta, start_x, start_y, start_theta, goal_x, goal_y, goal_theta), queue_id, start_node))
    
    path_draw = []
    collision_free = []
    collided = []
    open_set = set()
    closed_set = set()
    parents = {}
    g_cost = {}

    open_set.add(start_node)
    parents[start_node] = None

    while not q.empty() and iter < 10000:
        iter += 1
        current_priority, current_id, current_node = q.get()
        current_x, current_y, current_theta, current_g_cost = current_node.xytg()

        # If the heuristic between current position and goal is smaller than threshold, we find the path
        if(heuristic(current_x, current_y, current_theta, goal_x, goal_y, goal_theta) <= threshold):
            while current_node:
                path.append((current_node.x, current_node.y, current_node.theta))
                path_draw.append((current_node.x, current_node.y, 0.2))
                current_node = parents[current_node]
            path.reverse()
            success = True
            break

        closed_set.add(current_node)
        # open_set.remove(current_node)

        for dx, dy in directions:
            for angle in angles:
                queue_id += 1
                neighbor_x = current_x + dx * step_size
                neighbor_y = current_y + dy * step_size
                neighbor_theta = angle
                neighbor_g_cost = current_g_cost + action_cost(current_x, current_y, current_theta, neighbor_x, neighbor_y, neighbor_theta)
                neighbor_node = Node(neighbor_x, neighbor_y, neighbor_theta, neighbor_g_cost)
                
                if neighbor_node in closed_set:
                    continue

                if collision_fn((neighbor_x, neighbor_y, neighbor_theta)):
                    collided.append((neighbor_x, neighbor_y, 0.1))
                    # draw_sphere_marker((neighbor_x, neighbor_y, 0.2), 0.03, (1, 0, 0, 0.5))
                    continue
                
                # draw_sphere_marker((neighbor_x, neighbor_y, 0.2), 0.03, (0, 0, 1, 0.5))
                collision_free.append((neighbor_x, neighbor_y, 0.1))
                
                if (neighbor_node not in open_set) or (neighbor_g_cost < g_cost.get(neighbor_node, float('inf'))):
                    f_cost = total_cost(current_g_cost, current_x, current_y, current_theta, neighbor_x, neighbor_y, neighbor_theta, goal_x, goal_y, goal_theta)
                    queue_id += 1
                    q.put((f_cost, queue_id, neighbor_node))
                    open_set.add(neighbor_node)
                    parents[neighbor_node] = current_node
                    g_cost[neighbor_node] = neighbor_g_cost

    if success:
        total_path_cost = 0
        for i in range(len(path) - 1):
            x1, y1, theta1 = path[i]
            x2, y2, theta2 = path[i+1]
            total_path_cost += action_cost(x1, y1, theta1, x2, y2, theta2)
        print("Solution Found. Total Path Cost = ", total_path_cost)
    if not success:
        print("No Solution Found")

    

    # Draw the computed path in black
    # for path_points in path_draw:
    #     draw_sphere_marker(path_points, sphere_radius, (0, 0, 0, 1)) 
    for i in range(len(path_draw) - 1):
        draw_line(path_draw[i], path_draw[i+1], 5, (0, 0, 0))

    # Draw the x and y components of the collision-free configurations in blue
    existed_col_free = set()
    filtered_col_free = []
    for collision_free_points in collision_free:
        x, y, z = collision_free_points
        if (x, y) not in existed_col_free:
            existed_col_free.add((x, y))
            draw_sphere_marker(collision_free_points, sphere_radius, (0, 0, 1, 1))
            # filtered_col_free.append((x, y, z))
    
    # for i in range(len(filtered_col_free) - 1):
    #     draw_line(filtered_col_free[i], filtered_col_free[i+1], 3, (0, 0, 1))
            # draw_sphere_marker(collision_free_points, sphere_radius, (0, 0, 1, 1)) 

    # Draw the x and y components of the colliding configurations in red
    existed_col = set()
    # filtered_col = []
    # for collided_points in collided:
    #     x, y, z = collided_points
    #     if (x, y) not in existed_col:
    #         existed_col.add((x, y))
    #         filtered_col.append((x, y, z))
    
    # for i in range(len(filtered_col) - 1):
    #     draw_line(filtered_col[i], filtered_col[i+1], 3, (1, 0, 0))

    for collided_points in collided:
        x, y, z = collided_points
        if (x, y) not in existed_col:
            existed_col.add((x, y))
            draw_sphere_marker(collided_points, sphere_radius, (1, 0, 0, 1)) 
    
    ######################
    print("Planner run time: ", time.time() - start_time)
    # Execute planned path
    execute_trajectory(robots['pr2'], base_joints, path, sleep=0.2)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()