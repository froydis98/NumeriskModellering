import numpy as np

# Task 1
# Bisection method
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
            print(c)
        return c
    else:
        return "Not inside that interval"

def f(x):
    return x**2 - np.exp(-2*x)
    
print(bisection(f, 0.4, 0.8))
