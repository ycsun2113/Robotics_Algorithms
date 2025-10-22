import numpy as np

alpha = np.pi/2
beta = -np.pi/5
gamma = np.pi

# (i) A rotation of pi/2 about the z-axis
R1 = np.array([[np.cos(alpha), -np.sin(alpha), 0], 
               [np.sin(alpha), np.cos(alpha), 0], 
               [0, 0, 1]])

# (ii) A rotation of -pi/5 about the new y-axis
R2 = np.array([[np.cos(beta), 0, np.sin(beta)], 
               [0, 1, 0], 
               [-np.sin(beta), 0, np.cos(beta)]])

# (iii) A rotation of pi about the new z-axis
R3 = np.array([[np.cos(gamma), -np.sin(gamma), 0], 
               [np.sin(gamma), np.cos(gamma), 0], 
               [0, 0, 1]])

R = np.matmul(np.matmul(R1, R2), R3)

print('R:')
np.set_printoptions(precision=3, suppress=True)
print(R)