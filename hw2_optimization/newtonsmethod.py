import numpy as np
import backtracking as bt

'''
given a starting point x in dom(f), tolerance epsilon > 0

repeat
    1. Compute the Newton step and decrement
        delta_x_nt := - hess_f(x)^(-1) * grad_f(x);
        lambda^2 := trans(grad_f(x)) * hess_f(x)^(-1) * grad_f(x)
    2. Stopping criterion. quit if lambda^2/2 <= epsilon
    3. Line Search. Choose step size t by backtracking line search
    4. Update. x := x + t * delta_x_nt
'''
def newtonsmethod(x, f, grad_f, hess_f, alpha=0.1, beta=0.6, epsilon=0.0001):

    xi = [x]
    f_xi = [f(x)]
    nums_iter = 0

    while(True):
        # 1. Compute the Newton step and decrement
        # delta_x_nt := - hess_f(x)^(-1) * grad_f(x);
        # lambda^2 := trans(grad_f(x)) * hess_f(x)^(-1) * grad_f(x)
        delta_x_nt = -1 * (1/hess_f(x)) * grad_f(x)
        sqr_lambda = grad_f(x) * (1/hess_f(x)) * grad_f(x)

        # 2. Stopping criterion. quit if lambda^2/2 <= epsilon
        if(sqr_lambda/2 <= epsilon):
            break

        # 3. Line Search. Choose step size t by backtracking line search
        t = bt.backtracking(x, delta_x_nt, f, grad_f, alpha, beta)

        # 4. Update. x := x + t * delta_x_nt
        x = x + t * delta_x_nt

        # Append Results
        xi.append(x)
        f_xi.append(f(x))
        nums_iter += 1

    return xi, f_xi, nums_iter