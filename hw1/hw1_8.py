import numpy as np

# (a) Calculate the vector v in the world frame that the camera should align with
T = np.array([[np.sqrt(2)/2, np.sqrt(2)/2, 0, 1.7], 
              [-1*np.sqrt(2)/2, np.sqrt(2)/2, 0, 2.1], 
              [0, 0, 1, 0], 
              [0, 0, 0, 1]])

p = np.transpose(np.array([-0.4, 0.9, 0]))
current_pos = T[:3, 3]
# p = np.array([[-0.4], [0.9], [0]])
# current_pos = np.array([[1.7], [2.1], [0]])
vector = p - current_pos
vec_norm = vector / np.linalg.norm(vector)
# print(vector)

np.set_printoptions(precision=4, suppress=True)
print("(a) The transpose of vector v is:")
print(vector)
print()
print("(a) The transpose of normalized vector v is:")
print(vec_norm)
print()


# (b) Use v to calculate the desired pose of the robot
z_axis = np.array([0, 0, 1])
y_axis = np.cross(z_axis, vec_norm)
y_norm = y_axis / np.linalg.norm(y_axis)
x_axis = np.cross(y_norm, z_axis)

R = np.transpose(np.array([x_axis, y_axis, z_axis]))

print("(b) The rotation matrix should be: ")
print(R)
print()

desired_pose = np.zeros((4,4))
desired_pose[:3, :3] = R
desired_pose[:4, 3] = T[:4, 3]
print("(b) The desired pose of the robot is:")
print(desired_pose)
print()


# (c) Validfy the rotation matrix

# check if it is orthogonal and the determinant(R) == 1
if(np.allclose(np.matmul(R, np.transpose(R)), np.identity(3)) and (np.linalg.det(R) == 1)):
    print("(c) The rotation matrix is valid because it is orthogonal and the determinant(R) = 1.")
elif(not np.allclose(np.matmul(R, np.transpose(R)), np.identity(3))):
    print("(c) The rotation matrix is not valid because it is not orthogonal.")
elif(np.linalg.det(R) != 1):
    print("(c) The rotation matrix is not valid becuase Det(R) is not 1.")
