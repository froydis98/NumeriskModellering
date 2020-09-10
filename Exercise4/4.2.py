import numpy as np
import math

n = 10
A = np.random.rand(n, n) * 10
ANormalized = A.transpose() + A
values, vectors = np.linalg.eig(ANormalized)

def myPower(T, Niter):
    xVector = np.random.rand(n) * 10
    HighesVal = 0
    index = 0
    for i in range(len(T)):
        if HighesVal < T[i]:
            HighesVal = T[i]
            index = i
    for _ in range(Niter):
        xVector = xVector @ ANormalized
        xVector = xVector/abs(math.sqrt(xVector@xVector))
    correspondingVector = []
    for i in range(len(T)):
        correspondingVector.append(abs(vectors[i][index]))
    if (np.around(correspondingVector) == np.around(xVector)).all():
        return HighesVal, xVector
    else: 
        return 'Fant ingen'

print(myPower(values, 3))
