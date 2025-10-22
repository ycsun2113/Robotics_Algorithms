import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx

hyperplanes = np.asmatrix([[0.7071,    0.7071, 1.5], 
                    [-0.7071,    0.7071, 1.5],
                    [0.7071,    -0.7071, 1],
                    [-0.7071,    -0.7071, 1]])

#the optimization function c:
c = np.asmatrix([2, 1]).T

A = hyperplanes[:,0:2]
b = np.asarray(hyperplanes[:,2].flatten())
b = b.reshape(-1)


x = cvx.Variable(2)
prob = cvx.Problem(cvx.Minimize(c.T @ x),
                   [A @ x <= b])
prob.solve()

# print(f'The optimal point: ({x.value[0], x.value[1]})')
print("The optimal point: (%.2f, %.2f)"%(x.value[0], x.value[1]))