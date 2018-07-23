import numpy as np
from ypstruct import structure

# Run Differential EVolution
def run(problem, params):
    
    # Problem Definition
    objfunc = problem.objfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # Params
    maxit = params.maxit
    npop = params.npop
    F = params.F
    CR = params.CR
    DisplayInfo = params.DisplayInfo

    # Empty Individual
    empty_individual = structure()
    empty_individual.position = None
    empty_individual.cost = None

    # Best Solution
    bestsol = empty_individual.deepcopy()
    bestsol.cost = np.inf

    # Best Costs History
    bestcosts = np.empty(maxit)

    # Initialization
    pop = empty_individual.repeat(npop)
    for i in range(0, npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = objfunc(pop[i].position)
        if pop[i].cost <= bestsol.cost:
            bestsol = pop[i].deepcopy()

    # Evolution Loop
    for it in range(0, maxit):
        for i in range(0, npop):
            
            # Select Individuals
            A = select_individuals(i, npop)
            a = pop[A[0]]
            b = pop[A[1]]
            c = pop[A[2]]

            # Mutation
            y = apply_bounds(a.position + F * (b.position - c.position), varmin, varmax)

            # Crossover
            newsol = empty_individual.deepcopy()
            newsol.position = crossover(pop[i].position, y, CR)
            newsol.cost = objfunc(newsol.position)

            if newsol.cost <= pop[i].cost:
                pop[i] = newsol
                if pop[i].cost <= bestsol.cost:
                    bestsol = pop[i].deepcopy()

        # Store Best Cost of Iteration
        bestcosts[it] = bestsol.cost

        # Show Iteration Info
        if DisplayInfo:
            print("Iteration {0}: Best Cost = {1}".format(it, bestsol.cost))
    
    # Return Results
    out = structure()
    out.bestsol = bestsol
    out.bestcost = bestsol.cost
    out.bestcosts = bestcosts
    out.pop = pop
    return out

# Select Individuals for Crossover/Mutation
def select_individuals(i, npop):
    A = np.arange(npop)
    np.random.shuffle(A)
    A = np.delete(A, np.where(A == i))
    return A[0:3]

# Apply Decision Variable Ranges
def apply_bounds(x, varmin, varmax):
    x = np.maximum(x, varmin)
    x = np.minimum(x, varmax)
    return x

# Crossover
def crossover(x, y, CR):
    z = np.copy(x)
    
    nvar = x.size
    j = np.where(np.random.rand(nvar) <= CR)
    j = np.append(j, np.random.randint(0, nvar))
    z[j] = y[j]

    return z
    