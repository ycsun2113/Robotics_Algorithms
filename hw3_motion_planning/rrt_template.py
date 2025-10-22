import numpy as np
from utils import load_env, get_collision_fn_PR2, execute_trajectory
from pybullet_tools.utils import connect, disconnect, wait_if_gui, joint_from_name, get_joint_positions, set_joint_positions, get_joint_info, get_link_pose, link_from_name
import random
### YOUR IMPORTS HERE ###
from utils import draw_sphere_marker, draw_line
import time

JOINT_NAMES = ('l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_elbow_flex_joint', 'l_upper_arm_roll_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint')
STEP_SIZE = 0.05
GOAL_BIAS = 0.1
THRESHOLD = 0.5
DOF = len(JOINT_NAMES)


def sample_node(goal_config, joint_limits):
    pd = random.random()
    if pd <= GOAL_BIAS:
        return goal_config
    else:
        rand_nodes = []
        for jn in JOINT_NAMES:
            rand_nodes.append(joint_limits[jn][0] + random.random() * (joint_limits[jn][1] - joint_limits[jn][0]))
    return tuple(rand_nodes)
        
def distance(node1, node2):
    # origin: dist = np.linalg.norm((np.array(node1) - np.array(node2)) )
    dist = np.linalg.norm((np.array(node1) - np.array(node2)) * [0.5, 1.5, 0.7, 0.7, 0.7, 0.3])
    return dist

def nearest_neighbor(tree, rand_nodes):
    # dist = []
    # for node in tree:
    #     dist.append(distance(node, rand_nodes))
    # return tree[np.argmin(dist)]
    return min(tree, key=lambda node: distance(node, rand_nodes))
    
def new_val(nodes, joint_limits):
    for i in range(DOF):
        if (nodes[i] <= joint_limits[JOINT_NAMES[i]][0] or nodes[i] >= joint_limits[JOINT_NAMES[i]][1]) and i!=4:
            return False
    return True

def rrt_step(near_nodes, rand_nodes, joint_limits, collision_fn):
    direction = (np.array(rand_nodes) - np.array(near_nodes)) / distance(rand_nodes, near_nodes)
    new_nodes = [0] * DOF
    for i in range(DOF):
        if abs(rand_nodes[i] - near_nodes[i]) <= STEP_SIZE:
            new_nodes[i] = rand_nodes[i]
        else:
            new_nodes[i] = near_nodes[i] + direction[i] * STEP_SIZE
    return tuple(new_nodes) if not collision_fn(new_nodes) and new_val(new_nodes, joint_limits) else None

def construct_path(parents, node):
    path = []
    while node in parents:
        path.append(node)
        node = parents[node]
    return path

# def smoothing(path, collision_fn, joint_limits):
#     for i in range(150):
#         rd1, rd2 = random.sample(range(len(path)), 2)

#         if rd2 < rd1:
#             tmp = rd2
#             rd2 = rd1
#             rd1 = tmp
        
#         cut1 = path[rd1]
#         cut2 = path[rd2]
#         new_path = inter_path(cut1, cut2)
#         smoothed_path = path

#         for node in new_path:
#             if collision_fn(node) and not new_val(node, joint_limits):
#                 break
#             else:
#                 smoothed_path = smoothed_path[:rd1+1] + new_path + smoothed_path[rd2:]
#     return tuple(smoothed_path)    
    
# def inter_path(node1, node2):
#     direction = (np.array(node1) - np.array(node2)) / distance(node1, node2)
#     steps = (distance(node1, node2) / STEP_SIZE)
#     new_path = [0] * steps
#     for i in range(steps):
#         new_path[i] = node1[i] + i * direction * STEP_SIZE
#     return tuple(new_path)

    
#########################


joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')

def main(screenshot=False):
    # initialize PyBullet
    connect(use_gui=True)
    # load robot and obstacle resources
    robots, obstacles = load_env('pr2table.json')

    # define active DoFs
    joint_names =('l_shoulder_pan_joint','l_shoulder_lift_joint','l_elbow_flex_joint','l_upper_arm_roll_joint','l_forearm_roll_joint','l_wrist_flex_joint')
    joint_idx = [joint_from_name(robots['pr2'], jn) for jn in joint_names]

    # parse active DoF joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robots['pr2'], joint_idx[i]).jointLowerLimit, get_joint_info(robots['pr2'], joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}

    collision_fn = get_collision_fn_PR2(robots['pr2'], joint_idx, list(obstacles.values()))
    # Example use of collision checking
    # print("Robot colliding? ", collision_fn((0.5, 1.19, -1.548, 1.557, -1.32, -0.1928)))

    start_config = tuple(get_joint_positions(robots['pr2'], joint_idx))
    goal_config = (0.5, 0.33, -1.548, 1.557, -1.32, -0.1928)
    path = []
    ### YOUR CODE HERE ###

    rrt = [start_config]
    parents = {start_config: None}
    for i in range(100000):
        rand_node = sample_node(goal_config, joint_limits)
        near_nodes = nearest_neighbor(rrt, rand_node)
        new_nodes = rrt_step(near_nodes, rand_node, joint_limits, collision_fn)
        if new_nodes:
            rrt.append(new_nodes)
            parents[new_nodes] = near_nodes
            near_nodes = new_nodes
            if distance(new_nodes, goal_config) <= THRESHOLD:
                path = construct_path(parents, new_nodes)
                path.reverse()
                break
    else:
        print("No path found.")
        path = []

    for node in path:
        set_joint_positions(robots['pr2'], joint_idx, node)
        ee_pose = get_link_pose(robots['pr2'], link_from_name(robots['pr2'], 'l_gripper_tool_frame'))
        draw_sphere_marker(ee_pose[0], 0.04, (1, 0, 0, 1))


    # smoothed_path = smoothing(path, collision_fn, joint_limits)

    # for node in smoothed_path:
    #     set_joint_positions(robots['pr2'], joint_idx, node)
    #     ee_pose = get_link_pose(robots['pr2'], link_from_name(robots['pr2'], 'l_gripper_tool_frame'))
    #     draw_sphere_marker(ee_pose[0], 0.04, (0, 0, 1, 1))


    ######################
    # Execute planned path
    execute_trajectory(robots['pr2'], joint_idx, path, sleep=0.1)
    # Keep graphics window opened
    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()