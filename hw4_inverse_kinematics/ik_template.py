import numpy as np
from pybullet_tools.utils import connect, disconnect, set_joint_positions, wait_if_gui, set_point, load_model,\
                                 joint_from_name, link_from_name, get_joint_info, HideOutput, get_com_pose, wait_for_duration
from pybullet_tools.transformations import quaternion_matrix
from pybullet_tools.pr2_utils import DRAKE_PR2_URDF
import time
import sys
### YOUR IMPORTS HERE ###

#########################

from utils import draw_sphere_marker

def get_ee_transform(robot, joint_indices, joint_vals=None):
    # returns end-effector transform in the world frame with input joint configuration or with current configuration if not specified
    if joint_vals is not None:
        set_joint_positions(robot, joint_indices, joint_vals)
    ee_link = 'l_gripper_tool_frame'
    pos, orn = get_com_pose(robot, link_from_name(robot, ee_link))
    res = quaternion_matrix(orn)
    res[:3, 3] = pos
    return res

def get_joint_axis(robot, joint_idx):
    # returns joint axis in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    R_W_J = H_W_J[:3, :3]
    joint_axis_local = np.array(j_info.jointAxis)
    joint_axis_world = np.dot(R_W_J, joint_axis_local)
    return joint_axis_world

def get_joint_position(robot, joint_idx):
    # returns joint position in the world frame
    j_info = get_joint_info(robot, joint_idx)
    jt_local_pos, jt_local_orn = j_info.parentFramePos, j_info.parentFrameOrn
    H_L_J = quaternion_matrix(jt_local_orn) # joint transform in parent link CoM frame
    H_L_J[:3, 3] = jt_local_pos
    parent_link_world_pos, parent_link_world_orn = get_com_pose(robot, j_info.parentIndex)
    H_W_L = quaternion_matrix(parent_link_world_orn) # parent link CoM transform in world frame
    H_W_L[:3, 3] = parent_link_world_pos
    H_W_J = np.dot(H_W_L, H_L_J)
    j_world_posi = H_W_J[:3, 3]
    return j_world_posi

def set_joint_positions_np(robot, joints, q_arr):
    # set active DOF values from a numpy array
    q = [q_arr[0, i] for i in range(q_arr.shape[1])]
    set_joint_positions(robot, joints, q)


def get_translation_jacobian(robot, joint_indices, current_ee_position):
    J = np.zeros((3, len(joint_indices)))
    ### YOUR CODE HERE ###

    # print("====================================")
    # print("Running get_translation_jabobian()...")
    # print("joint_indices:")
    # print(joint_indices)
    p = np.zeros((3, len(joint_indices)))

    # for a translation (prismatic) joint: dx/dq_i = v_i (joint axis)
    # for a rotation (revolute) joint: dx/dq_i = v_i * p_i
    # p_i = from joint center to the end-effector pose
    for i in range(len(joint_indices)):
        # print("joint_inidices[i]:")
        # print(joint_indices[i])
        current_joint_position = get_joint_position(robot, joint_indices[i])
        # print("current_joint_position:")
        # print(current_joint_position)
        # print("current_ee_position:")
        # print(current_ee_position)
        v_i = get_joint_axis(robot, joint_indices[i])
        p_i = current_ee_position - current_joint_position
        # print("v_i:")
        # print(v_i)
        # print("p_i:")
        # print(p_i)
        J[:, i] = np.cross(v_i, p_i)

    # print("J:")
    # print(J)

    ### YOUR CODE HERE ###
    return J

def get_jacobian_pinv(J):
    J_pinv = []
    ### YOUR CODE HERE ###

    # print("====================================")
    # print("Running get_jacobian_pinv()...")
    # Right Pseudo-Inverse of the Jocobian:
    # J^+ = J^T * (J * J^T + lambda^2 * I)^-1
    const_lambda = 0.01
    I = np.identity((J@np.transpose(J)).shape[0])
    J_pinv = np.transpose(J) @ np.linalg.inv(J @ np.transpose(J) + (const_lambda * const_lambda * I))
    # print("J_pinv:")
    # print(J_pinv)


    ### YOUR CODE HERE ###
    return J_pinv

def tuck_arm(robot):
    joint_names = ['torso_lift_joint','l_shoulder_lift_joint','l_elbow_flex_joint',\
        'l_wrist_flex_joint','r_shoulder_lift_joint','r_elbow_flex_joint','r_wrist_flex_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    set_joint_positions(robot, joint_idx, (0.24,1.29023451,-2.32099996,-0.69800004,1.27843491,-2.32100002,-0.69799996))

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        print("Specify which target to run:")
        print("  'python3 ik_template.py [target index]' will run the simulation for a specific target index (0-4)")
        exit()
    test_idx = 0
    try:
        test_idx = int(args[0])
    except:
        print("ERROR: Test index has not been specified")
        exit()

    # initialize PyBullet
    connect(use_gui=True, shadows=False)
    # load robot
    with HideOutput():
        robot = load_model(DRAKE_PR2_URDF, fixed_base=True)
        set_point(robot, (-0.75, -0.07551, 0.02))
    tuck_arm(robot)
    # define active DoFs
    joint_names =['l_shoulder_pan_joint','l_shoulder_lift_joint','l_upper_arm_roll_joint', \
        'l_elbow_flex_joint','l_forearm_roll_joint','l_wrist_flex_joint','l_wrist_roll_joint']
    joint_idx = [joint_from_name(robot, jn) for jn in joint_names]
    # intial config
    q_arr = np.zeros((1, len(joint_idx)))
    set_joint_positions_np(robot, joint_idx, q_arr)
    # list of example targets
    targets = [[-0.15070158,  0.47726995, 1.56714123],
               [-0.36535318,  0.11249,    1.08326675],
               [-0.56491217,  0.011443,   1.2922572 ],
               [-1.07012697,  0.81909669, 0.47344636],
               [-1.11050811,  0.97000718,  1.31087581]]
    # define joint limits
    joint_limits = {joint_names[i] : (get_joint_info(robot, joint_idx[i]).jointLowerLimit, get_joint_info(robot, joint_idx[i]).jointUpperLimit) for i in range(len(joint_idx))}
    q = np.zeros((1, len(joint_names))) # start at this configuration
    target = targets[test_idx]
    # draw a blue sphere at the target
    draw_sphere_marker(target, 0.05, (0, 0, 1, 1))
    
    ### YOUR CODE HERE ###
    
    # print(joint_limits)
    def FK(robot, joint_idx, joint_vals):
        # print("joint_vals:")
        # print(joint_vals)
        ee_transform = get_ee_transform(robot, joint_idx, joint_vals)
        current_ee_position = ee_transform[:3, 3]
        return current_ee_position
    
    threshold = 0.01
    alpha = 0.01
    max_iter = 500
    # set initial q = {0, 0, 0, 0, 0, 0, 0, 0}
    q_initial = np.zeros((1, len(joint_idx)))
    q_current = [q_initial[0, i] for i in range(q_initial.shape[1])]
    x_target = target
    print("====================================")
    print("q starting at:")
    print(q_initial)
    # set_joint_positions(robot, joint_idx, q_current)
    print("target end-effector position:")
    print(x_target)

    for i in range(max_iter):
        # q_current: current_joint_position
        # x_current: current_ee_position = FK(current_joint_position)
        # q_current = get_joint_position(robot, joint_idx)
        x_current = FK(robot, joint_idx, q_current)

        # x_dot = (x_target = x_current)
        x_dot = (x_target - x_current)

        # error = ||x_dot||
        error = np.linalg.norm(x_dot)

        # if error<threshold
        #     return q_current
        if(error < threshold):
            break
        
        # q_dot = J(q)^+ * x_dot
        J = get_translation_jacobian(robot, joint_idx, x_current)
        J_pinv = get_jacobian_pinv(J)
        q_dot = J_pinv @ x_dot

        # if(||q_dot|| > alpha)
        #     q_dot = alpha(q_dot / ||q_dot||)
        if(np.linalg.norm(q_dot) > alpha):
            q_dot = alpha * (q_dot / np.linalg.norm(q_dot))
        
        # q_current = q_current + q_dot
        q_current = q_current + q_dot

        # print("q_current at iter = ", i)
        # print(q_current)

        # print("====================================")
        # print("Check joint limits begins...")
        # print("q_current:")
        # print(q_current)
        # print("joint_limits:")
        # print(joint_limits)

        for i in range(len(joint_idx)):
            if i == 4 or i == 6:
                continue
            if q_current[i] < joint_limits[joint_names[i]][0]:
                q_current[i] = joint_limits[joint_names[i]][0]
            elif q_current[i] > joint_limits[joint_names[i]][1]:
                q_current[i] = joint_limits[joint_names[i]][1]
            else:
                pass
    
    print("====================================")
    print("Done!")
    print("final configuration:")
    print(q_current)
    print("final end-effector position:")
    print(x_current)
    print("====================================")


    ### YOUR CODE HERE ###

    wait_if_gui()
    disconnect()

if __name__ == '__main__':
    main()