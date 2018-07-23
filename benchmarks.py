import numpy as np

# Sphere
def sphere(x):
    return sum(x**2)

# Rastrigin
def rastrigin(x):
    A = 10
    n = x.size
    return A*n + sum(x**2 - A*np.cos(2*np.pi*x))

# Rosenbrock
def rosenbrock(x):
    return sum(100*(x[1:] - x[:-1]**2)**2 + (1-x[:-1])**2)

# Ackley
def ackley(x):
    a = 20
    b = 0.2
    c = 2*np.pi
    return a + np.e - a*np.exp(-b*np.mean(x**2)) - np.exp(np.mean(np.cos(c*x)))
