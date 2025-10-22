import numpy as np
import matplotlib.pyplot as plt
import backtracking as bt
import gradientdescent as gd
import newtonsmethod as nt

def f(x):
    return np.exp(0.5*x + 1) + np.exp(-0.5*x - 0.5) + 0.5*x

def grad_f(x):
    return 0.5*np.exp(0.5*x + 1) + (-0.5)*np.exp(-0.5*x - 0.5) + 0.5

def hess_f(x):
    return 0.25*np.exp(0.5*x + 1) + 0.25*np.exp(-0.5*x - 0.5)

# Starting point:
x_0 = 5

# Parameters:
alpha = 0.1
beta = 0.6
epsilon = 0.0001

# Gradient Descent Method:
gd_xi, gd_yi, gd_iter = gd.gradientdescent(x_0, f, grad_f, alpha, beta, epsilon)

# Newton's Method
nt_xi, nt_yi, nt_iter = nt.newtonsmethod(x_0, f, grad_f, hess_f, alpha, beta, epsilon)

# Print their number of iterations
print(f'Gradient Descent Method\'s number of iterations: {gd_iter}')
print(f'Newton\'s Method\'s number of iterations: {nt_iter}')

# Create Plots
xvals = np.arange(-10, 10, 0.01) # Grid of 0.01 spacing from -10 to 10
yvals = f(xvals) # Evaluate function on xvals

# Objective function and the sequence of points
plt.figure()
plt.plot(xvals, yvals, 'k-', label='Objective Function') # Create line plot with yvals against xvals
plt.plot(gd_xi, gd_yi, 'ro-', label='Gradient Descent')
plt.plot(nt_xi, nt_yi, 'mo-', label='Newton\'s Method')
plt.xlabel('xi')
plt.ylabel('f(xi)')
plt.title('Descent Methods')
plt.legend()
plt.grid()
plt.show() #show the plot

# f(x(i)) vs i
plt.figure()
plt.plot(gd_yi, 'ro-', label='Gradient Descent')
plt.plot(nt_yi, 'mo-', label='Newton\'s Method')
plt.xlabel('i')
plt.ylabel('f(x(i))')
plt.title('f(x(i)) vs i')
plt.legend()
plt.grid()
plt.show()



