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
    highestEigenvalue = abs(xv @ T @ xv)
    return xv, highestEigenvalue

vec, val = myPower(ANormalized, 10)
A2 = ANormalized - (val * np.outer(vec, vec))
x = np.around(A2 @ vec)
vec2, val2 = myPower(A2, 10)
A3 = A2 - (val2 * np.outer(vec2, vec2))

