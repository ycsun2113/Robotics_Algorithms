#!/usr/bin/env python
import utils
import numpy as np
import time
import random
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import random
from pca_template import PCA_rotate_denoise
from ransac_template import Fit
import ransac_template as ransac

def Error(point, plane):
    x, y, z = point
    a, b, c, d = plane
    num = abs(a*x + b*y + c*z + d)
    den = np.sqrt(a*a + b*b + c*c)
    if den == 0:
        dist = np.nan
    else:
        dist = num / den
    return dist

###YOUR IMPORTS HERE###

def add_some_outliers(pc,num_outliers):
    pc = utils.add_outliers_centroid(pc, num_outliers, 0.75, 'uniform')
    random.shuffle(pc)
    return pc

def main():
    #Import the cloud
    pc = utils.load_pc('cloud_pca.csv')

    num_tests = 10
    fig = None
    fig2 = None
    fig3 = None
    pca_error_list = []
    pca_runtime_list = []
    ransac_error_list = []
    ransac_runtime_list = []
    outliers_num = []
    iter_num = []
    for i in range(0,num_tests):
        pc = add_some_outliers(pc,10) #adding 10 new outliers for each test
        # fig = utils.view_pc([pc])

        ###YOUR CODE HERE###
        
        # numbers of iterations:
        K = 7000
        # threshold of inliers:
        DELTA = 0.25
        # minimum number of consensus points required:
        # N = 150
        N = len(pc) * 0.8

        print("=============================================")
        print(f"iter = {i+1}")
        outliers_num.append((i+1) * 10)
        iter_num.append((i+1))
        mu = np.mean(pc, axis=0)

        ### PCA ###
        print("")
        print("Running PCA...")
        pca_start_time = time.time()
        pc_new, U, SIGMA, VT, pc_new_s, VsT = PCA_rotate_denoise(pc)
        pca_time = time.time() - pca_start_time

        V = np.transpose(VT)
        S = SIGMA*SIGMA
        print("S:")
        print(S)
        # PCA threshold:
        pca_threshold = min(S) * 1.1
        
        V_notimportant = V[:, S < pca_threshold]
        pca_normal = []
        for k in range(3):
            pca_normal.append((V_notimportant[k,0]))
        a, b, c = pca_normal
        pca_normal = np.matrix([a, b, c]).reshape(3,1)
        x, y, z = np.squeeze(mu)
        d = -(a*x + b*y + c*z)
        pca_plane = [a, b, c, d]
        print("PCA plane:")
        print(f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        pca_inliers = []
        pca_outliers = []
        for point in pc:
            if Error(point, pca_plane) < DELTA:
                pca_inliers.append((np.asarray(point)))
            else:
                pca_outliers.append((np.asarray(point)))

        # pca_error = sum(Error(pt, pca_plane)**2 for pt in pca_outliers)
        pca_error = sum(Error(pt, pca_plane)**2 for pt in pca_inliers)
        # print(pca_error[0])
        print(f"PCA error = {pca_error[0]:.4f}")
        print(f"PCA runtime = {pca_time:.4f}")
        pca_error_list.append(pca_error)
        pca_runtime_list.append(pca_time)

        ### RANSAC ###
        print("")
        print("Running RANSAC...")
        ransac_start_time = time.time()
        ransac_plane, error_best = ransac.RANSAC(pc, K, DELTA, N, False)
        ransac_time = time.time() - ransac_start_time

        a, b, c, d = ransac_plane
        ransac_normal = np.matrix([a, b, c]).reshape(3,1)
        print("RANSAC plane:")
        print(f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
        
        ransac_inliers = []
        ransac_outliers = []
        for point in pc:
            if Error(point, ransac_plane) < DELTA:
                ransac_inliers.append((np.asarray(point)))
            else:
                ransac_outliers.append((np.asarray(point)))

        # ransac_error = sum(Error(pt, ransac_plane)**2 for pt in ransac_outliers)
        ransac_error = sum(Error(pt, ransac_plane)**2 for pt in ransac_inliers)
        print(f"RANSAC error = {ransac_error[0]:.4f}")
        print(f"RANSAC runtime = {ransac_time:.4f}")
        ransac_error_list.append(ransac_error)
        ransac_runtime_list.append(ransac_time)

        ### Plot ###
        # fig2 = utils.view_pc([pca_outliers])
        # fig2 = utils.view_pc([pca_inliers], fig2, 'r')
        # utils.draw_plane(fig2, pca_normal, mu, (0, 1.0, 0, 0.2))

        # fig3 = utils.view_pc([ransac_outliers])
        # fig3 = utils.view_pc([ransac_inliers], fig3, 'r')
        # utils.draw_plane(fig3, ransac_normal, mu, (0, 1.0, 0, 0.2))

        #this code is just for viewing, you can remove or change it
        # input("Press enter for next test:")
        plt.close(fig)
        # plt.close(fig2)
        # plt.close(fig3)
        ###YOUR CODE HERE###

    ### Plot ###
    print("==============================================")
    print("Plotting PCA's plane (Figure 2)")
    fig2 = utils.view_pc([pca_outliers])
    fig2 = utils.view_pc([pca_inliers], fig2, 'r')
    utils.draw_plane(fig2, pca_normal, mu, (0, 1.0, 0, 0.2))
    plt.title("PCA's plane")

    print("Plotting RANSAC's plane (Figure 3)")
    fig3 = utils.view_pc([ransac_outliers])
    fig3 = utils.view_pc([ransac_inliers], fig3, 'r')
    utils.draw_plane(fig3, ransac_normal, mu, (0, 1.0, 0, 0.2))
    plt.title("RANSAC's plane")

    print("Plotting Error vs. Number of Outliers (Figure 4)")
    plt.figure(4)
    plt.plot(outliers_num, pca_error_list, label = 'PCA', color = 'r')
    plt.plot(outliers_num, ransac_error_list, label = 'RANSAC', color = 'b')
    plt.title("Error vs. Number of Outliers (PCA vs. RANSAC)")
    plt.xlabel("Number of Outliers")
    plt.ylabel("Error")
    plt.legend()
    
    print("Plotting Error vs. Number of Outliers (only PCA) (Figure 5)")
    plt.figure(5)
    plt.plot(outliers_num, pca_error_list, label = 'PCA', color = 'r')
    plt.title("Error vs. Number of Outliers (PCA)")
    plt.xlabel("Number of Outliers")
    plt.ylabel("Error")

    print("Plotting Error vs. Number of Outliers (only RANSAC) (Figure 6)")
    plt.figure(6)
    plt.plot(outliers_num, ransac_error_list, label = 'RANSAC', color = 'b')
    plt.title("Error vs. Number of Outliers (RANSAC)")
    plt.xlabel("Number of Outliers")
    plt.ylabel("Error")

    print("Plotting Computation Time at Each Iteration (Figure 7)")
    plt.figure(7)
    plt.plot(iter_num, pca_runtime_list, label = 'PCA', color = 'r')
    plt.plot(iter_num, ransac_runtime_list, label = 'RANSAC', color = 'b')
    plt.title("Computation Time at Each Iteration (PCA vs. RANSAC)")
    plt.xlabel("Iteration")
    plt.ylabel("Computation Time")
    plt.legend()

    print("Plotting Computation Time at Each Iteration (only PCA) (Figure 8)")
    plt.figure(8)
    plt.plot(iter_num, pca_runtime_list, label = 'PCA', color = 'r')
    plt.title("Computation Time at Each Iteration (PCA)")
    plt.xlabel("Iteration")
    plt.ylabel("Computation Time")

    print("Plotting Computation Time at Each Iteration (only RANSAC) (Figure 9)")
    plt.figure(9)
    plt.plot(iter_num, ransac_runtime_list, label = 'RANSAC', color = 'b')
    plt.title("Computation Time at Each Iteration (RANSAC)")
    plt.xlabel("Iteration")
    plt.ylabel("Computation Time")

    plt.show()

    input("Press enter to end")


if __name__ == '__main__':
    main()
