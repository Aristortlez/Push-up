# import required libraries
import numpy as np
from numpy.linalg import norm
 
# define two lists or array
A = np.array([-21.05, -22.66, -87.91, -6.23, 7.58, 20.21])
B = np.array([-21.678589139614346, -23.244983847244356, -85.91438322002513, -5.679249716384954, 0, 26.56505117707799])
 
print("A:", A)
print("B:", B)
 
# compute cosine similarity
cosine = np.dot(A,B)/(norm(A)*norm(B))
print("Cosine Similarity:", cosine)