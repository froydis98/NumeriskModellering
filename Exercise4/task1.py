import numpy as np

n = 10
A = np.random.rand(n, n) * 10
ANormalized = A.transpose() + A
values, vectors = np.linalg.eig(ANormalized)
Orthonormal = np.zeros((n, n))
for i in range(len(vectors)):
    for j in range(len(vectors)):
        Orthonormal[i,j] = abs(np.around(vectors[j] @ vectors[i]))
print(Orthonormal)
P = vectors
PTrans = P.transpose()
Diagonalized = np.around(PTrans @ ANormalized @ P)
print(Diagonalized)
print(np.around(values))



