import numpy as np
import matplotlib.pyplot as plt
import time
import random
import SGDtest
import sgd
import gradientdescent as gd
import newtonsmethod as nt

# Q2(c)
def run_sgd(num=30, iters=1000):

    fsum_opt = []
    
    for i in range(num):
        xi, x_opt = sgd.sgd(step_size=1, max_iters=iters, x_0=-5)
        fsum_opt.append(SGDtest.fsum(x_opt))

    mean = np.mean(fsum_opt)
    var = np.var(fsum_opt)
    return mean, var

mean_1000, var_1000 = run_sgd(30, 1000)
mean_750, var_750 = run_sgd(30, 750)

print('Q2(c)')
print(f'750 Iterations: mean = {mean_750}, variance = {var_750}')
print(f'1000 Iterations: mean = {mean_1000}, variance = {var_1000}')


# Q2.(d).(i). Compare the runtimes for the three algorithms

# Starting point:
x_0 = 5

# Parameters:
alpha = 0.1
beta = 0.6
epsilon = 0.0001

# SGD
sgd_start = time.time()
sgd_xi, sgd_xopt = sgd.sgd(step_size=1, max_iters=1000, x_0=x_0)
sgd_end = time.time()

# Gradient Descent Method:
gd_start = time.time()
gd_xi, gd_yi, gd_iter = gd.gradientdescent(x_0, SGDtest.fsum, SGDtest.fsumprime, alpha, beta, epsilon)
gd_end = time.time()

# Newton's Method
nt_start = time.time()
nt_xi, nt_yi, nt_iter = nt.newtonsmethod(x_0, SGDtest.fsum, SGDtest.fsumprime, SGDtest.fsumprimeprime, alpha, beta, epsilon)
nt_end = time.time()


# Q2.(d).(ii). Compare the three algorithms in terms of fsum(x*)
sgd_result = SGDtest.fsum(sgd_xopt)
gd_result = SGDtest.fsum(gd_xi[-1])
nt_result = SGDtest.fsum(nt_xi[-1])

# Print the result for Q2(d)
print()
print('Q2(d)')
print('Stochastic Gradient Descent:')
print('Runtime: ', sgd_end - sgd_start)
print('fsum(x*): ', sgd_result)

print()
print('Gradient Descent:')
print('Runtime: ', gd_end - gd_start)
print('fsum(x*): ', gd_result)

print()
print('Newton\'s Method:')
print('Runtime: ', nt_end - nt_start)
print('fsum(x*): ', nt_result)
