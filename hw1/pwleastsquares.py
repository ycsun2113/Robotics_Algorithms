import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("calibration.txt")
commanded = data[:, 0]
measured = data[:, 1]
knots = [-0.5, 0.5]

A = np.zeros((len(commanded), 4))
A[:, 0] = 1
A[:, 1] = commanded
A[:, 2] = np.maximum(0, commanded - knots[0])
A[:, 3] = np.maximum(0, commanded - knots[1])

A_inv = np.linalg.pinv(A)


x = np.matmul(A_inv, measured)
print(f"Parameters: {x}")

fitted = np.matmul(A, x)

err = np.sum((measured - fitted) ** 2)
print(f"The sum of squared errors: {err}")

command = 0.68
A_command_pred = np.array([1, command, np.maximum(0, command - knots[0]), np.maximum(0, command - knots[1]), ])
result = np.matmul(A_command_pred, x)
print(f"The predicted measured position when command = 0.68 will be {result}.")

plt.plot(commanded, measured, 'bx', label='Measured')
plt.plot(commanded, fitted, 'r', label='Fitted Line')
plt.xlabel('Commanded Position')
plt.ylabel('Measured Position')
plt.title('Piece-Wise Linear Least-Squares Fit')
plt.legend()
plt.show()


