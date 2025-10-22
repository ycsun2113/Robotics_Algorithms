import numpy as np

# Problem 5(a)
A_a = np.array([[0, 0, -1], [4, 1, 1], [-2, 2, 1]])
B_a = np.array([[3], [1], [1]])
det_A = np.linalg.det(A_a)

if(det_A != 0):
    x_a = np.linalg.solve(A_a, B_a)
    print("x_a:")
    print(x_a)
else:
    print("A_a is not invertible, there will be no solution or have infinitely many solutions for (a).")

# Problem 5(b)
A_b = np.array([[0, -2, 6], [-4, -2, -2], [2, 1, 1]])
B_b = np.array([[1], [-2], [0]])
det_A = np.linalg.det(A_b)

if(det_A != 0):
    x_b = np.linalg.solve(A_b, B_b)
    print("x_b:")
    print(x_b)
else:
    print("A_b is not invertible, there will be no solution or have infinitely many solutions for (b).")

# Problem 5(c)
A_c = np.array([[2, -2], [-4, 3]])
B_c = np.array([[3], [-2]])
det_A = np.linalg.det(A_c)

if(det_A != 0):
    x_c = np.linalg.solve(A_c, B_c)
    print("x_c:")
    print(x_c)
else:
    print("A_c is not invertible, there will be no solution or have infinitely many solutions for (c).")