import numpy as np
import matplotlib.pyplot as plt

def Kepler(r, M):
    return np.sqrt(r**3/M)

def plot():
    list1, list2 = [], []
    for x in np.arange(0, 2, 0.1):
        list1.append(Kepler(x,1)) 
        list2.append(Kepler(x,2))
    plt.xlim(0, 10)
    plt.ylim(0, 1.2)
    plt.plot(list1)
    plt.plot(list2)
    plt.savefig('kepler_figure.pdf')
    plt.show()

def newPlot():
    r = np.arange(0, 10, 0.1)
    plt.plot(r, Kepler(r, 1))
    plt.plot(r, Kepler(r, 2))
    plt.show()
plot()
