#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
def distance(pi, qi):
    dist = np.linalg.norm(pi - qi)
    return dist

def closest_point(pi, pc_target):
    dist = []
    for qi in pc_target:
        dist.append(distance(pi, qi))
    closest_idx = np.argmin(dist)
    return pc_target[closest_idx]

def get_transform(Cp, Cq):
    # print("Cp shape:")
    # print(np.shape(Cp))

    source_pts = Cp
    target_pts = Cq
    # print("source_pts shape:")
    # print(np.shape(source_pts))

    # Compute means
    p_mean = np.mean(source_pts, axis=0)
    q_mean = np.mean(target_pts, axis=0)

    # Compute centered vectors
    X = source_pts - p_mean
    Y = target_pts - q_mean
    X = np.reshape(np.array(X), (len(Cp),3))
    Y = np.reshape(np.array(Y), (len(Cp),3))

    # print("X, Y shape:")
    # print(np.shape(X))
    # print(np.shape(Y))

    # Compute SVD of covariance matrix of centered vectors
    S = np.transpose(X) @ Y
    
    U, SIGMA, VT = np.linalg.svd(S)

    # Compute R
    V = np.transpose(VT)
    UT = np.transpose(U)
    M = np.array([[1, 0, 0], 
                  [0, 1, 0], 
                  [0, 0, np.linalg.det(V @ UT)]])
    R = np.matmul(np.matmul(V, M), UT)

    # Compute t
    t = np.asarray(q_mean - R @ p_mean)

    return R, t


def Error(Cp, Cq, R, t):
    transformed_Cp = np.matmul(R, Cp) + t
    obj = transformed_Cp - Cq
    err = sum((obj[i][0]*obj[i][0]+obj[i][1]*obj[i][1]+obj[i][2]*obj[i][2]) for i in range(len(Cp)))
    return err


###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc_source = utils.load_pc('cloud_icp_source.csv')

    ###YOUR CODE HERE###
    # default:
    pc_target = utils.load_pc('cloud_icp_target0.csv')

    # print("input a target number (0~3):")
    target_num = int(input("input a target number (0~3): "))
    epsilon = 0.0001
    max_iter = 100
    if target_num == 0:
        epsilon = 0.015312
        max_iter = 30
        pc_target = utils.load_pc('cloud_icp_target0.csv') # Change this to load in a different target
    elif target_num == 1:
        epsilon = 0.0001
        max_iter = 20
        pc_target = utils.load_pc('cloud_icp_target1.csv')
    elif target_num == 2:
        epsilon = 0.008
        max_iter = 60
        pc_target = utils.load_pc('cloud_icp_target2.csv')
    elif target_num == 3:
        epsilon = 0.04
        max_iter = 50
        pc_target = utils.load_pc('cloud_icp_target3.csv')
    
    # Origin Point Cloud
    fig1 = None
    fig1 = utils.view_pc([pc_source, pc_target], fig1, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])

    iter_num = []
    error_per_iter = []
    # fig1 = None

    for i in range(max_iter):
        # print(f"iter = {i+1}")
        # if i%5==0:
        #     plt.close(fig1)
        #     fig1 = utils.view_pc([pc_source, pc_target], None, ['b', 'r'], ['o', '^'])
        #     plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])
        iter_num.append(i+1)

        # Compute Correspondences
        Cp = []
        Cq = []
        for pi in pc_source:
            qi = closest_point(pi, pc_target)
            Cp.append(pi)
            Cq.append(qi)
        
        # Compute Transform
        R, t = get_transform(Cp, Cq)

        error = Error(Cp, Cq, R, t)
        error = np.squeeze(error).item(0)
        error_per_iter.append(error)
        print(f"iter = {i+1}, error = {error:.8f}")
        if  error < epsilon:
            for i in range(len(Cp)):
                pc_source[i] = R @ pc_source[i] + t
            break

        # Update all P
        for i in range(len(Cp)):
            pc_source[i] = R @ pc_source[i] + t
        
    fig2 = None
    fig2 = utils.view_pc([pc_source, pc_target], fig2, ['b', 'r'], ['o', '^'])
    plt.axis([-0.15, 0.15, -0.15, 0.15, -0.15, 0.15])

    plt.figure(3)
    plt.plot(iter_num, error_per_iter, color='blue')
    plt.title("Error vs. iteration of ICP")
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    # plt.show()


    ###YOUR CODE HERE###

    plt.show()
    #raw_input("Press enter to end:")


if __name__ == '__main__':
    main()
