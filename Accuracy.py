import numpy as np

# กำหนดเวกเตอร์ vec1 และ vec3
vec1 = np.array([-21.05 ,-22.66 ,-87.91  ,-6.23   ,7.58  ,20.21])
vec3 = np.array([0, -20, -50, 0, 0, 0])

# คำนวณ norm ของเวกเตอร์ vec1 และ vec3
norm_vec1 = np.linalg.norm(vec1)
norm_vec3 = np.linalg.norm(vec3)
print('norm_v1',norm_vec1)
print('norm_v1',norm_vec3)

# แปลงเวกเตอร์เป็น normalized vector
v1 = vec1 / norm_vec1
v3 = vec3 / norm_vec3
print('vec1/norm_vec1',v1)
print('vec3/norm_vec3',v3)

# คำนวณ 1 - norm ของผลต่างของเวกเตอร์ที่ normalized
result = (1 - np.linalg.norm(v1 - v3)) * 100 

print("Accuracy", result)