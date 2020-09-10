import numpy as np
import math

n = 10
A = np.random.rand(n, n) * 10
ANormalized = A.transpose() + A
values, vectors = np.linalg.eig(ANormalized)

def myPower(T, Niter):
    xv = np.random.rand(n) * 10
    for _ in range(Niter):
        xv = xv @ T
        xv = xv/abs(math.sqrt(xv@xv))
    highestEigenvalue = xv @ T @ xv
    return xv, highestEigenvalue

print(myPower(ANormalized, 10))
