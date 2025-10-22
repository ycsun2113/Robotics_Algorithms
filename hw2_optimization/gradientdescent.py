import numpy as np
import backtracking as bt

'''
given a starting point x in dom(f)

repeat
    1. Determine a descent direction delta_x
    2. Line search. Choose a step size t>0
    3. Update. x := x + t * delta_x

until stopping criterion is satisfied
'''


def gradientdescent(x, f, grad_f, alpha=0.1, beta=0.6, epsilon=0.0001):
    
    xi = [x]
    f_xi = [f(x)]
    nums_iter = 0

    # Repeat until stopping criterion is satisfied
    while(np.linalg.norm(grad_f(x)) > epsilon):
        # 1. Determine a descent direction delta_x
        delta_x = -grad_f(x)

        # 2. Line Search. Choose a step size t>0
        t = bt.backtracking(x, delta_x, f, grad_f, alpha, beta)

        # 3. Update. x := x + t * delta_x
        x = x + t * delta_x

        # Append Results
        xi.append(x)
        f_xi.append(f(x))
        nums_iter += 1

    return xi, f_xi, nums_iter
