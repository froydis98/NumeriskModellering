import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
from scipy import optimize

def f(x):
    return (1.0 + np.sin(x * np.pi) - 2.0 * x)

def plot():
    i = np.arange(0, 2, 0.1)
    plt.plot(i, f(i), 'r')
    plt.plot(i, np.gradient(f(i)), 'b')
    plt.grid()
    plt.show()

def df(x):
    return (np.pi*np.cos(np.pi*x) - 2.0)

def newtonFor(x):
    for i in range(10):
        h = f(x)/df(x)
        x = x - h
        print(x)
    return x

def myNewton(f, x, df):
    h = f(x) / df(x)
    while abs(h) >= 0.0001:
        h = (f(x)/df(x))
        x -= h
    return x

print(myNewton(f, 0.1, df))

print(optimize.newton(f, 0.1))
