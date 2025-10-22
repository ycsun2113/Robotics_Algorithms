#!/usr/bin/env python
import utils
import numpy as np
import matplotlib.pyplot as plt
###YOUR IMPORTS HERE###
import random

# def random_subset(points):
#     # idx = np.random.choice(len(points), 3, replace=False)
#     # return [np.asarray(points[i]) for i in idx]
#     R = []
#     for i in range(3):
#         idx = random.randint(0, len(points)-1)
#         R.append(np.squeeze(np.asarray(points[idx])))
#     return R

def Fit(points):
    # if len(points) == 3:
    #     p1 = points[0]
    #     p2 = points[1]
    #     p3 = points[2]
    # else:
    #     indices = []
    #     used = np.full((len(points)), False)
    #     for j in range(3):
    #         idx = random.randint(0, len(points)-1)
    #         while used[idx] == True:
    #             idx = random.randint(0, len(points)-1)
    #         indices.append(idx)
    #         used[idx] = True
    #     p1 = points[indices[0]]
    #     p2 = points[indices[1]]
    #     p3 = points[indices[2]]

    indices = []
    pt_exist = np.full((len(points)), False)
    for j in range(3):
        idx = random.randint(0, len(points)-1)
        while pt_exist[idx] == True:
            idx = random.randint(0, len(points)-1)
        indices.append(idx)
        pt_exist[idx] = True
    p1 = points[indices[0]]
    p2 = points[indices[1]]
    p3 = points[indices[2]]

    v1 = p2 - p1
    v2 = p3 - p1

    normal_vec = np.cross(v1, v2)
    a, b, c = normal_vec
    d = -np.dot(normal_vec, p1)
    plane = [a, b, c, d]
    return plane

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


def RANSAC(pc, K, DELTA, N, print_update = True):
    error_best = float('inf')

    for i in range(K):
        # if i%50 == 0:
        #     print(f"iter = {i}")
        # 1. Pick a random subset R in P
        # R = random_subset(pc)
        R = []
        for m in range(3):
            idx = random.randint(0, len(pc)-1)
            R.append(np.squeeze(np.asarray(pc[idx])))
        # 2. theta = Fit(model, R)
        plane = Fit(R)
        # 3. C = {0}
        C = []
        # 4. For (p in P\R):
        #        if (error(p, model(theta)) < delta):
        #            C = C U p
        for pt in pc:
            pt = np.squeeze(np.asarray(pt))
            if any(np.array_equal(pt, r) for r in R):
                continue
            if Error(pt, plane) < DELTA:
                C.append(pt)
        # 5. if (|C| > N):
        #        theta = Fit(model, R U C)
        #        error_new = error(R U C, model(theta))
        #        if (error_new < error_best):
        #            error_best = error_new
        #            theta_best = theta
        if len(C) > N:
            R_C = R + C
            new_plane = Fit(R_C)
            error_new = 0
            error_new = sum(Error(pt, new_plane)**2 for pt in R_C)
            # for pt in R_C:
            #     error_new = error_new + Error(pt, new_plane)
            if error_new < error_best:
                error_best = error_new
                plane_best = new_plane
                inliers_best = C
                if print_update == True:
                    print(f"iter = {i}, err_best = {error_best:.4f}")

    return plane_best, error_best

###YOUR IMPORTS HERE###


def main():
    #Import the cloud
    pc = utils.load_pc('cloud_ransac.csv')


    ###YOUR CODE HERE###
    # Show the input point cloud
    fig1 = utils.view_pc([pc])

    #Fit a plane to the data using ransac

    # numbers of iterations:
    K = 3000
    # threshold of inliers:
    DELTA = 0.2
    # minimum number of consensus points required:
    N = len(pc) * 0.5

    plane_best, error_best= RANSAC(pc, K, DELTA, N, True)
    a, b, c, d = plane_best
    # normal_vec = np.array([a, b, c])
    # normal_vec = normal_vec[:, None]
    normal_vec = np.matrix([a, b, c]).reshape(3,1)
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    # print(np.shape(normal_vec))
    print("===============================================")
    print("best error:")
    print(error_best)
    print("best plane:")
    print(f"{a:.4f} x + {b:.4f} y + {c:.4f} z + {d:.4f} = 0")
    # print(np.shape(pt))
    # print(pt.reshape(3,1))
    
    #Show the resulting point cloud
    inliers = []
    outliers = []
    for point in pc:
        if Error(point, plane_best) < DELTA:
            inliers.append((np.asarray(point)))
        else:
            outliers.append((np.asarray(point)))
    
    # print(np.shape(pc))
    # print(np.shape(inliers))

    fig2 = utils.view_pc([outliers])
    fig2 = utils.view_pc([inliers], fig2, 'r')

    #Draw the fitted plane
    # utils.draw_plane(fig2, normal_vec, pt.reshape(3,1), (0, 1.0, 0, 0.2))
    mu = np.mean(pc, axis = 0)
    utils.draw_plane(fig2, normal_vec, mu, (0, 1.0, 0, 0.2), [-0.75, 1.0], [-1.25, 1.25])

    ###YOUR CODE HERE###
    plt.show()
    # input("Press enter to end:")


if __name__ == '__main__':
    main()
