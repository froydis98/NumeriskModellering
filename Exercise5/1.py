import numpy as np
from matplotlib import pyplot as plt

N = 100
x = np.arange(0, 1, N)
xi = np.random.randn(N)

mean = np.sum(xi)/N

