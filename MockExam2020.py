import numpy as np
from scipy.linalg import lu, inv
import math
import matplotlib.pylab as plt

# Task 1
def secant(f, x0, x1, eps_r, eps_a):
    listOfX = []
    while True:
        x = x1 - (f(x1) * (x1 - x0)) / (f(x1) - f(x0))
        listOfX.append(x)
        if abs(x1 - x) < eps_a or abs((x1 - x)/x) < eps_r:
            return listOfX, len(listOfX)
        if x != x:
            return 'The secant line does not intersect with the x-axis, try to use other start values.'
        x0 = x1
        x1 = x

def fMock(x):
    return x**2 - np.exp(-2*x)

def dfMock(x):
    return 2*x + 2*np.exp(-2*x)

print(secant(fMock, 0.4, 0.6, 0.001, 0.001))

def bisection(f, a, b):
    if(f(a)*f(b) < 0):
        c = a
        while ((b-a) >= 0.0000001):
            c = (a+b)/2
            if(f(c) == 0.0):
                break
            if(f(c) * f(a) < 0):
                b = c
            else:
                a = c
        return c
    else:
        return "Not inside that interval"

print(bisection(fMock, 0.4, 0.8))


# Task 2

A = [[2, -1, 5], [1, 1, -3], [2, 4, 1]]
b = [10, -2, 1]

# Finding the condition number
Atest = [[2, -1, 1], [1, 0, 1], [3, -1, 4]]
print(np.linalg.cond(A, 1))
print(np.linalg.cond(A, np.inf))

# Gauss Elemination
def myGauss(A, b):
    Combined = np.column_stack((A, b))
    Height = len(Combined)
    Width = len(Combined[0])

    # Gaussian Elemination
    for i in range(Height):
        for j in range(i+1, Height):
            x = Combined[j][i]/Combined[i][i]
            for k in range(Width):
                Combined[j][k] = Combined[j][k] - x * Combined[i][k]
                
    solved = []
    for i in range(Height-1, -1, -1):
        solved.append(Combined[i][Height])
    # Back substitution 
    for y in range(Height-1, -1, -1):
        tmp = 0
        for z in range(y+1, Height):
            tmp += Combined[y][z]*solved[z]
        solved[y] = (Combined[y, Height]-tmp)/Combined[y][y]
    return solved

print(myGauss(A, b))
print(np.linalg.solve(A, b))

# Find inverse matrix
print(np.linalg.inv(A))

# Find the LU-decomposition
P, L, U = lu(A)
print(L)
print(U)
print(P)

# Find least squared
B=[[3, 1], [1, -2], [1, -1], [2, 3]]
c=[5, -3, 1, 6] 
leastSquared = np.linalg.lstsq(B, c, rcond=0.01)
print(leastSquared)


# Task 3
Anp = np.array(A)
print("Characteristic equation: ", np.poly(A))
print("Eigenvals:", np.linalg.eigvals(A))
ANormalized = Anp.transpose() + Anp
values, vectors = np.linalg.eig(ANormalized)
print("Eigenvectors: ", vectors)
print("Eigenvalues: ", values)

def myPower(T, Niter):
    xv = np.random.rand(3) * 3
    for _ in range(Niter):
        xv = xv @ T
        xv = xv/abs(math.sqrt(xv@xv))
    highestEigenvalue = abs(xv @ T @ xv)
    return xv, highestEigenvalue

vec, val = myPower(ANormalized, 3)
print(vec, val)

# Deflate the largest eigenvalue of matrix A
Z = vec.transpose() * vec * val
Anew = A - Z
vec2, val2 = myPower(Anew, 3)
print(val2)

# Task 4

x=[1900, 1950, 1980, 1990, 2000, 2010]
y=[400, 550, 980, 1130, 1270, 1390]
xsum = 0
ysum = 0
xsquared = 0
xy = 0

for i in range (len(x)):
    xsum += x[i]
    ysum += y[i]
    xsquared += x[i]*x[i]
    xy += x[i]*y[i]
print(xy)
M = (len(x)*xy-xsum*ysum)/(len(x)*xsquared-xsum**2)
b = (ysum-M*xsum)/(len(x))
print(M, b)




# Task 7

