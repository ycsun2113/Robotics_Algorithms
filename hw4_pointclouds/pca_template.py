#!/usr/bin/env python
import utils
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
def PCA_rotate_denoise(pc):
    # 1. Given a dataset X
    # 2. Compute the mean of X, mu
    mu = np.mean(pc, axis = 0)

    # print("mean:")
    # print(mu)
    # 3. X = X - mu (substract mu from every point in X)
    pc = utils.convert_pc_to_matrix(pc)
    pc = pc - mu
    # 4. SVD(Q) = U*sigma*V^T
    n = pc.shape[1]
    covariance_Q = (pc * np.transpose(pc)) / (n - 1)
    U, SIGMA, VT = np.linalg.svd(covariance_Q)
    # 5. Compute X_new = V^T * X
    pc_new = VT * pc

    # Compute the variance of each principle component
    S = SIGMA * SIGMA
    # print("S:")
    # print(S)
    # threshold = 0.00005
    threshold = min(S) * 1.1
    
    # Remove columns of V shose corresponding entry in s in less than some small threshold
    V = np.transpose(VT)
    Vs = V[:, S > threshold]
    # print("V:")
    # print(V)
    # print("Vs:")
    # print(Vs)
    VsT = np.transpose(Vs)
    pc_new_s = VsT * pc
    
    return pc_new, U, SIGMA, VT, pc_new_s, VsT

###YOUR IMPORTS HERE###


def main():

    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    ###YOUR CODE HERE###
    ### Show the input point cloud ###
    print("Plotting the input point cloud (Figure 1)")
    fig = utils.view_pc([pc])

    #### Rotate the points to align with the XY plane ###
    print("=====================================")
    print("Rotating the points to align with the XY plane...")
    pc_origin = pc
    # 1. Given a dataset X
    # 2. Compute the mean of X, mu
    mu = np.mean(pc, axis = 0)
    # print("mean:")
    # print(mu)
    # 3. X = X - mu (substract mu from every point in X)
    pc = utils.convert_pc_to_matrix(pc)
    pc = pc - mu
    # 4. Compute Covariance Q = (X*X^T)/(n-1)
    n = pc.shape[1]
    covariance_Q = (pc * np.transpose(pc)) / (n - 1)
    # 5. SVD(Q) = U*sigma*V^T
    U, SIGMA, VT = np.linalg.svd(covariance_Q)
    # 6. Compute X_new = V^T * X
    pc_new = VT * pc
    
    print("VT:")
    print(VT)

    ### Show the resulting point cloud ###
    print("Plotting the point cloud aligns with the XY plane (Figure 2)")
    pc_new = utils.convert_matrix_to_pc(pc_new)
    fig2 = utils.view_pc([pc_new])
    

    ### Rotate the points to align with the XY plane AND eliminate the noise ###
    print("=====================================")
    print("Eliminating the noise...")
    # 6. pc_new_s, VsT = PCA_denoise(pc, U, SIGMA, VT)
    S = SIGMA * SIGMA
    # print("S:")
    # print(S)
    # threshold = 0.00005
    threshold = min(S) * 1.5
    print(f"threshold = {threshold}")
    
    # 7. Remove columns of V shose corresponding entry in s in less than some small threshold
    V = np.transpose(VT)
    Vs = V[:, S > threshold]
    VsT = np.transpose(Vs)
    # print(np.shape(VsT))
    pc_new_s = VsT * pc

    print("VsT:")
    print(VsT)

    ### Show the resulting point cloud ###
    print("Plotting the point cloud aligns with the XY plane and all the z values are 0 (Figure 3)")
    # print(np.shape(pc_new_s))
    pc_new_s = np.vstack([pc_new_s, np.zeros((1, 200))])
    # print(np.shape(pc_new_s))
    pc_new_s = utils.convert_matrix_to_pc(pc_new_s)
    fig3 = utils.view_pc([pc_new_s])

    ### Fit a plane ##
    print("=====================================")
    print("Fitting a plane...")
    V_notimportant = V[:, S < threshold]
    normal_vec = V_notimportant[:, 0]
    print("normal vector of the plane:")
    print(normal_vec)
    # print(np.shape(pc_origin))
    print("Plotting the plane with the point cloud (Figure 4)")
    fig4 = utils.view_pc([pc_origin])
    # print(np.shape(mu))
    utils.draw_plane(fig4, normal_vec, mu, (0, 1.0, 0, 0.2))


    ###YOUR CODE HERE###

    plt.show()
    #input("Press enter to end:")


if __name__ == '__main__':
    main()
