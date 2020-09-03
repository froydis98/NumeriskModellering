import numpy as np

A = np.array([[1., 2, 3], [4, 1, 3], [5, 1, 2]])
b = np.array([1., 2, 3])

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
