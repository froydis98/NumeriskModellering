import numpy as np

n = 10
A = np.random.rand(n, n) * 10
ANormalized = A.transpose() + A
values, vectors = np.linalg.eig(ANormalized)
R = np.zeros((n, n))
for i in range(len(vectors)):
    for j in range(len(vectors)):
        R[i,j] = abs(np.around(vectors[j] @ vectors[i]))

P = vectors
PTrans = P.transpose()
D = np.around(PTrans@ANormalized@P)
print(D)
print(np.around(values))



