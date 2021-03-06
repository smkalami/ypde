# Yarpiz Differential Evolution (ypde)
Differential Evolution in Python

## How to use?
A basic usage and sample application is implemented in `app.py`. You may choose any of benchmark functions defined in `benchmarks.py`. Before using this, you should install `numpy`, `matplotlib` and `ypstruct`.

```
# External Libraries
import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure

# From This Project
import de
import benchmarks

# Problem Definition
problem = structure()
problem.objfunc = benchmarks.rastrigin  # See benchmarks module for other functions
problem.nvar = 100
problem.varmin = -10
problem.varmax = 10

# Parameters of Differential Evolution (DE)
params = structure()
params.maxit = 1000
params.npop = 100
params.F = 0.2
params.CR = 0.4
params.DisplayInfo = True

# Run DE
out = de.run(problem, params)

# Print Final Result
print("Final Best Solution: {0}".format(out.bestsol))

# Plot of Best Costs History
plt.semilogy(out.bestcosts)
plt.xlim(0, params.maxit)
plt.xlabel("Iterations")
plt.ylabel("Best Cost")
plt.title("Differential Evolution")
plt.grid(True)
plt.show()
```