# -*- coding: utf-8 -*-
import math
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB
import random
from bokeh.plotting import figure, show


def tspmip(n, dist, timelimit=60):
    m = gp.Model()
    # Objects to use inside callbacks
    m._n = n
    m._subtours = []
    m._tours = []
    m._dist = dict(dist)

    # Create variables
    m._vars = vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='x')
    # Create opposite direction (i,j) -> (j,i)
    # This isn't a new variable - it's a pointer to the same variable
    for i, j in vars.keys():
        vars[j, i] = vars[i, j]
        m._dist[j, i] = dist[i, j]

    # Add degree-2 constraint
    m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))
    # Set parameter for lazy constraints
    m.Params.lazyConstraints = 1
    # Set the relative MIP gap to 0 and the time limit
    m.Params.MIPGap = 0
    m.Params.TimeLimit = timelimit
    # Set the absolute MIP gap to the smallest nonzero difference in distances
    distvals = sorted(dist.values())
    m.Params.MIPGapAbs = min(v[1]-v[0] for v in list(zip(distvals[:-1], distvals[1:])) if v[1] != v[0])
    return m


def subtours(vals):
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
    cycles = []
    while edges:
        # Trace edges until we find a loop
        i, j = edges[0]
        thiscycle = [i]
        while j != thiscycle[0]:
            thiscycle.append(j)
            i, j = next((i, j) for i, j in edges.select(j, '*') if j != thiscycle[-2])
        cycles.append(thiscycle)
        for j in thiscycle:
            edges.remove((i, j))
            edges.remove((j, i))
            i = j
    return sorted(cycles, key=lambda x: len(x))


def tourcost(dist, tour):
    return sum(dist[tour[k-1], tour[k]] for k in range(len(tour)))


def tspcb(heurcb=None):
    def collect_tours(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            tours = subtours(vals)
            if len(tours) > 1:
                model._subtours.append(tours)
            else:
                model._tours.append(tours[0])
                # Record time when first tour is found
                try:
                    model._firstsoltime
                except AttributeError:
                    model._firstsoltime = model.cbGet(GRB.Callback.RUNTIME)

    def lazy_cons(model, where):
        # Add subtour constraints if there are any subtours
        if where == GRB.Callback.MIPSOL:
            for tours in model._subtours:
                # add a subtour elimination constraint for all but largest subtour
                for tour in tours[:-1]:
                    model.cbLazy(gp.quicksum(
                        model._vars[i, j] for i, j in combinations(tour, 2) if (i, j) in model._vars) <= len(tour)-1)
            # Reset the subtours
            model._subtours = []

    def inject_sol(model, where):
        if where == GRB.Callback.MIPNODE:
            if model._tours:
                # There may be multiple tours - find the best one
                tour, cost = min(
                    ((tour, tourcost(model._dist, tour)) for tour in model._tours),
                    key=lambda x: x[-1])
                # Only apply if the tour is an improvement
                if cost < model.cbGet(GRB.Callback.MIPNODE_OBJBST):
                    # Set all variables to 0.0 - optional but helpful to suppress some warnings
                    model.cbSetSolution(model._vars.values(), [0.0]*len(model._vars))
                    # Now set variables in tour to 1.0
                    model.cbSetSolution([model._vars[tour[k-1], tour[k]] for k in range(len(tour))], [1.0]*len(tour))
                    # Use the solution - optional but a slight performance improvement
                    model.cbUseSolution()
                # Reset the tours
                model._tours = []

    def wrapper(model, where):
        collect_tours(model, where)
        if heurcb:
            heurcb(model, where)
        lazy_cons(model, where)
        inject_sol(model, where)
    return wrapper


def checksol(m, points, plot=True):
    print('')
    if m.SolCount > 0:
        vals = m.getAttr('x', m._vars)
        tours = subtours(vals)

        if len(tours) == 1:
            if m.Status == GRB.OPTIMAL:
                status = "Optimal TSP tour"
            else:
                status = "Suboptimal TSP tour"
            output = tours[0]
        else:
            status = "%i TSP subtours" % len(tours)
            output = tours
        print('%s: %s' % (status, str(output)))
        print('Cost: %g' % m.objVal)
        if plot:
            plotsol(points, tours, "%s on %i cities, length=%f" % (status, m._n, m.objVal))
    else:
        print('No solution!')
    print('')


def plotsol(points, tours, title="", path=False):
    fig = figure(title=title, x_range=[0, 100], y_range=[0, 100])
    x, y = zip(*points)
    fig.circle(x, y, size=8)
    for tour in tours:
        ptseq = [points[k] for k in tour]
        if not path:
            ptseq.append(ptseq[0])
        x, y = zip(*ptseq)
        fig.line(x, y)
    show(fig)


def addruntimes(runtimes, method, model):
    # remove old copy, if one exists
    try:
        i = runtimes['methods'].index(method)
        for rt in runtimes.values():
            rt.pop(i)
    except ValueError:
        pass
    # add new value
    runtimes['methods'].append(method)
    runtimes['optimal'].append(model.Runtime)
    try:
        runtimes['firstsol'].append(model._firstsoltime)
    except AttributeError:
        runtimes['firstsol'].append(model.Runtime)


def main():
    n = 60
    random.seed(1)
    points = [(random.randint(0, 100), random.randint(0, 100)) for i in range(n)]
    dist = {(i, j):
            math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
            for i in range(n) for j in range(i)}
    runtimes = {'methods': [], 'optimal': [], 'firstsol': []}
    m = tspmip(n, dist)
    m.optimize(tspcb())
    checksol(m, points)
    addruntimes(runtimes, 'noheur', m)


main()
import sys
sys.exit(1)


class pytsp:
    def __init__(self, n, dist, logging=False):
        self.n = n
        self.dist = dist
        self.logging = logging

    # Construct a heuristic tour via greedy insertion
    def greedy(self, dist=None, sense=1):
        if not dist:
            dist = self.dist
        unexplored = list(range(n))
        tour = []
        prev = 0
        while unexplored:
            best = min((i for i in unexplored if i != prev), key=lambda k: sense*dist[prev,k])
            tour.append(best)
            unexplored.remove(best)
            prev = best
        if self.logging:
            print("**** greedy heuristic tour=%f, obj=%f" % (tourcost(self.dist, tour), tourcost(dist, tour)))
        return tour

    # Construct a heuristic tour via Karp patching method from subtours
    def patch(self, subtours):
        if self.logging:
            print("**** patching %i subtours" % len(subtours))
        tours = list(subtours) # copy object to avoid destroying it
        while len(tours) > 1:
            # t1,t2 are tours to merge
            # k1,k2 are positions to merge in the tours
            # d is the direction - forwards or backwards
            t2 = tours.pop()
            # Find best merge
            j1, k1, k2, d, obj = min(((j1,k1,k2,d,
                                        self.dist[tours[j1][k1-1],  t2[k2-d]]      +
                                        self.dist[tours[j1][k1],    t2[k2-1+d]]    -
                                        self.dist[tours[j1][k1-1],  tours[j1][k1]] -
                                        self.dist[t2[k2-1],         t2[k2]])
                                      for j1 in range(len(tours))
                                      for k1 in range(len(tours[j1]))
                                      for k2 in range(len(t2))
                                      for d in range(2)), # d=0 is forward, d=1 is reverse
                                    key=lambda x: x[-1])
            t1 = tours[j1]
            k1 += 1 # include the position
            k2 += 1
            if d == 0: # forward
                tour = t1[:k1]+t2[k2:]+t2[:k2]+t1[k1:]
            else: # reverse
                tour = t1[:k1]+list(reversed(t2[:k2]))+list(reversed(t2[k2:]))+t1[k1:]
            tours[j1] = tour # replace j1 with new merge
        if self.logging:
            print("**** patched tour=%f" % tourcost(self.dist, tour))
        return tours[0]

    # Improve a tour via swapping
    # This is simple - just do 2-opt
    def swap(self, tour):
        if self.logging:
            beforecost = tourcost(self.dist, tour)

        for j1 in range(len(tour)):
            for j2 in range(j1+1, len(tour)):
                if self.dist[tour[j1-1],tour[j1]]+self.dist[tour[j2-1],tour[j2]] > \
                   self.dist[tour[j1-1],tour[j2-1]]+self.dist[tour[j1],tour[j2]]:
                    # swap
                    tour = tour[:j1] + list(reversed(tour[j1:j2])) + tour[j2:]

        if self.logging:
            print("**** swapping: before=%f after=%f" % (beforecost, tourcost(self.dist, tour)))
        return tour


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Try swap heuristic
# When a tour has been discovered in the MIP, call the swap heuristic to try and improve it.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Callback for swap heuristic
# Since the base callback injects a tour at a MIP node, this should be called at a MIP node.

# %% slideshow={"slide_type": "subslide"}
def swapcb(model, where):
    if where == GRB.Callback.MIPNODE:
        pt = pytsp(model._n, model._dist)
        for k in range(len(model._tours)):
            model._tours[k] = pt.swap(model._tours[k])


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Solve the TSP with the swap heuristic
# By itself, this should be no faster at finding the first solution, but it may reduce the time to optimality.

# %% slideshow={"slide_type": "subslide"}
m = tspmip(n, dist)
m.optimize(tspcb(swapcb))
checksol(m)
addruntimes(runtimes, 'swap', m)


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Try greedy heuristic
# - While solving the MIP, call the greedy heuristic using the _fractional values from the LP relaxation_
# - The motivation is that these fractional values should guide towards a good solution
# - When values are all zero (like crossing between subtours), pick the edge with the shortest length.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Callback for greedy heuristic

# %% slideshow={"slide_type": "subslide"}
def greedycb(model, where):
    if where == GRB.Callback.MIPNODE:
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x = model.cbGetNodeRel(model._vars)
            for k in x:
                if x[k] < 0.001:
                    x[k] = -model._dist[k]
            pt = pytsp(model._n, model._dist)
            model._tours.append(pt.greedy(dist=x, sense=-1)) # maximize using the x values


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Solve the TSP with the greedy heuristic

# %% slideshow={"slide_type": "subslide"} tags=[]
m = tspmip(n, dist)
m.optimize(tspcb(greedycb))
checksol(m)
addruntimes(runtimes, 'greedy', m)


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Try patch heuristic
# When an integer solution contains subtours, call the patching heuristic to create a tour, and try that as a heuristic solution.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Callback for patch heuristic

# %% slideshow={"slide_type": "subslide"}
def patchcb(model, where):
    if where == GRB.Callback.MIPSOL:
        pt = pytsp(model._n, model._dist)
        for subtour in model._subtours:
            model._tours.append(pt.patch(subtour))


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Solve the TSP with the patch heuristic

# %% slideshow={"slide_type": "subslide"} tags=[]
m = tspmip(n, dist)
m.optimize(tspcb(patchcb))
checksol(m)
addruntimes(runtimes, 'patch', m)


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Try fix-and-dive heuristic
# - When a fractional solution contains some variables at 1, try to fix those and solve the submodel
# - Although this is similar to a built-in MIP heuristic, this also calls the subtour callback inside.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Callback for fix-and-dive heuristic
# Note that this is also written as a closure. The reason for this is that we want to specify a heuristic callback function when solving the fixed model!

# %% slideshow={"slide_type": "subslide"}
def fixcb(subcb=None):
    def inner(model, where):
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                # Try solving the fixed submodel
                fixed = model._fixed
                # Relaxed values near 1.0 get the lower bound set to 1.0
                for k,v in model.cbGetNodeRel(model._vars).items():
                    fixed._vars[k].LB = math.floor(v+0.01)
                # Set a cutoff for the fixed model, based on the current best solution
                if model.cbGet(GRB.Callback.MIPNODE_SOLCNT) > 0:
                    fixed.Params.Cutoff = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                fixed.optimize(tspcb(subcb)) # call subproblem callback
                if fixed.status == GRB.OPTIMAL:
                    fixedvals = fixed.getAttr('x', fixed._vars)
                    model.cbSetSolution(model._vars, fixedvals)
    return inner


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Create the fixed model
#
# We create two copies of the model and disable output when solving the smaller fixed model.

# %% slideshow={"slide_type": "subslide"}
def tspmipwithfixed(n, dist):
    m = tspmip(n, dist) # main model
    m._fixed = tspmip(n, dist) # fixed model
    m._fixed.Params.OutputFlag = 0
    m._fixed._parent = m
    return m


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Solve the TSP with the fix-and-dive heuristic

# %% slideshow={"slide_type": "subslide"} tags=[]
m = tspmipwithfixed(n, dist)
m.optimize(tspcb(fixcb()))
checksol(m)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Tuning the callback for the fixed model
#
# - One issue is that the fixed values may be infeasible due to subtours
# - Let's exploit this by passing subtours found by the fixed model back to the parent model
#
# First, create a callback function _for the fixed model_ that appends the subtours to the subtours for parent model:

# %% slideshow={"slide_type": "subslide"}
def passfixsubtours(model, where):
    if where == GRB.Callback.MIPSOL:
        model._parent._subtours += model._subtours


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Solve the TSP with the enhanced fix-and-dive heuristic
# The callback function appears complicated!
#
# - What it's doing is to call the main callback (`tspcb`) on the MIP with the heuristic fixed callback `fixcb`
# - The fixed model uses the callback `passfixsubtours`, which sends subtours back to the original MIP.

# %% slideshow={"slide_type": "subslide"} tags=[]
m = tspmipwithfixed(n, dist)
m.optimize(tspcb(fixcb(passfixsubtours)))
checksol(m)


# %% [markdown] slideshow={"slide_type": "slide"}
# ## Multiple heuristics
# Why not combine multiple heuristics together?

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Combination: Greedy + Swap

# %% slideshow={"slide_type": "subslide"}
def combo(model, where):
    greedycb(model, where)
    swapcb(model, where)

m = tspmip(n, dist)
m.optimize(tspcb(combo))
checksol(m)
addruntimes(runtimes, 'GS', m)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Combination: Patch + Swap

# %% slideshow={"slide_type": "subslide"}
def combo(model, where):
    patchcb(model, where)
    swapcb(model, where)

m = tspmip(n, dist)
m.optimize(tspcb(combo))
checksol(m)
addruntimes(runtimes, 'PS', m)


# %% [markdown] slideshow={"slide_type": "slide"}
# ### Combination: Patch + Greedy + Swap

# %% slideshow={"slide_type": "subslide"}
def combo(model, where):
    patchcb(model, where)
    greedycb(model, where)
    swapcb(model, where)

m = tspmip(n, dist)
m.optimize(tspcb(combo))
checksol(m)
addruntimes(runtimes, 'PGS', m)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Performance
# Compare performance of the different heuristics

# %% slideshow={"slide_type": "subslide"}
from bokeh.transform import dodge

fig = figure(x_range=runtimes['methods'], title="Runtimes")
fig.vbar(x=dodge('methods', -0.2, range=fig.x_range),
         top='firstsol', source=runtimes, width=0.3, color="red",
         legend_label="First solution")
fig.vbar(x=dodge('methods', 0.2, range=fig.x_range),
         top='optimal', source=runtimes, width=0.3, color="blue",
         legend_label="Optimality")
show(fig)

# %% [markdown] slideshow={"slide_type": "slide"}
# # General model
# - The MIP TSP *does not* require a Euclidean distance function
# - It does not even require the triangle inequality!
#
# Let's try some purely random distances:

# %% slideshow={"slide_type": "subslide"}
n = 300

random.seed(20)
dist = {(i, j): random.uniform(0,100)
        for i in range(n) for j in range(i)}

m = tspmip(n, dist)

m.optimize(tspcb(patchcb))

checksol(m, plot=False)

# %% [markdown]
# ## Models that are likely to benefit from custom MIP heuristics
# - Where it is difficult to find integer solutions via the LP relaxation
# - Where it is easy to construct or improve an integer solution

# %% [markdown]
# ## Models that are unlikely to benefit from custom heuristics
# - Where it is easy to find integer solutions
# - Where default MIP heuristics perform well
#     - Ex: knapsack problems

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Example models that are likely to benefit from custom MIP heuristics
#
# - Models with some possibility
#   - Set covering/packing: Can you do better than general MIP rounding?
# - Promising models with disjunctive constraints
#   - Sequencing / disjunctive scheduling
#   - 2D/3D bin packing
#   - Open pit mining

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Remember the Disclaimers
# - We use the Traveling Salesman Problem (TSP) **for illustration purposes**
#   - Why TSP? Because it is a rich model that is easy to understand
# - This is *not* designed to show the fastest method for the TSP
#   - Special-purpose TSP codes outperform this model
#   - If you want to solve a TSP, consider a state-of-the-art system like [Concorde TSP Solver](http://www.math.uwaterloo.ca/tsp/concorde.html)

# %% [markdown] slideshow={"slide_type": "slide"}
# # How to get this code
# Available for download on Github: https://github.com/Gurobi/pres-mipheur
#
# **NOTE**: The sample data is too large to run using a _free trial license_; please do one of the following:
# - Commercial prospects: [Contact Gurobi sales](https://www.gurobi.com/company/contact-us/) to get a time-limited evaluation license
# - Academic users: Get a [free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/) (if you qualify)
# - Anyone: Reduce the value of n to get a smaller model instance

# %% [markdown] slideshow={"slide_type": "slide"}
# # Questions/Discussion

# %% slideshow={"slide_type": "skip"}
