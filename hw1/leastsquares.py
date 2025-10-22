import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("calibration.txt")
commanded = data[:, 0]
measured = data[:, 1]

np.set_printoptions(precision=3, suppress=True)

A = np.ones((len(commanded), 2))
A[:, 0] = commanded
b = measured

A_inv = np.linalg.pinv(A)

x = np.matmul(A_inv, b)

print("Parameters:")
print(f"X1(slope) = {x[0]}")
print(f"X2(intercept) = {x[1]}")

fitted = x[0]*commanded + x[1]
# print(fitted)

err = np.sum((measured - fitted) ** 2)
print(f"The sum of squared errors: {err}")

plt.plot(commanded, measured, 'bx', label='Measured')
plt.plot(commanded, fitted, 'r', label='Fitted Line')
plt.xlabel('Commanded Position')
plt.ylabel('Measured Position')
plt.title('Least-Squares Fit')
plt.legend()
plt.show()