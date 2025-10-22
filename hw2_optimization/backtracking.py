import numpy as np

### Backtracking Line Search
# Given a descent direction delta(x) for f at x in dom(f)
# alpha=0.1, beta=0.6
# f(x + t*delta(x)) > f(x) + alpha*t*gradient(f(x))^T*delta(x)

def backtracking(x, delta_x, f, grad_f, alpha=0.1, beta=0.6):
    t = 1
    while(f(x + t*delta_x) > f(x) + alpha * t * np.dot(grad_f(x), delta_x)):
       t = beta*t
    return t

