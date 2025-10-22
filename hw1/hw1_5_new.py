import numpy as np

def solve_x(A, b):
    rank_A = np.linalg.matrix_rank(A)
    augmented = np.column_stack((A, b))
    rank_aug = np.linalg.matrix_rank(augmented)
    
    if(rank_A == rank_aug):
        if(rank_A == A.shape[1]): # if A is an m*n matrix, A.shape[1] = n
            x = np.linalg.solve(A, b)
            print("x =")
            print(x)
        else:
            print("There is infinitely many solutions for this equation.")
    else:
        print("There is no solution for this equation.")

A_a = np.array([[0, 0, -1], [4, 1, 1], [-2, 2, 1]])
B_a = np.array([[3], [1], [1]])
print("(a):")
solve_x(A_a, B_a)

A_b = np.array([[0, -2, 6], [-4, -2, -2], [2, 1, 1]])
B_b = np.array([[1], [-2], [0]])
print("(b):")
solve_x(A_b, B_b)

A_c = np.array([[2, -2], [-4, 3]])
B_c = np.array([[3], [-2]])
print("(c):")
solve_x(A_c, B_c)