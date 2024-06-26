import numpy as np

# กำหนดเวกเตอร์ vec1 และ vec3
vec1 = np.array([-21.05, -22.66, -87.91, -6.23, 7.58, 20.21])
vec3 = np.array([-21.678589139614346, -23.244983847244356, -85.91438322002513, -5.679249716384954, 0, 26.56505117707799])

# คำนวณ norm ของเวกเตอร์ vec1 และ vec3
norm_vec1 = np.linalg.norm(vec1)
norm_vec3 = np.linalg.norm(vec3)

# แปลงเวกเตอร์เป็น normalized vector
v1 = vec1 / norm_vec1
v3 = vec3 / norm_vec3
print('vec1/norm_vec1',v1)
print('vec3/norm_vec3',v3)

# คำนวณ 1 - norm ของผลต่างของเวกเตอร์ที่ normalized
accuracy_percentage = (1 - np.linalg.norm(v1 - v3)) * 100

print("1 - Norm of the difference between v1 and v3:", accuracy_percentage)