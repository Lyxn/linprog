# Linear Programming 

## Introduction
Linear programming problem in standard form is
```
min  c * x
s.t. A * x = b,
     x >= 0.
```
And the dual problem is 
```
max r * b
s.t. r * A <= c
```
In the integer programming, the variables must be integers.

#### Artificial Variable
One can add artificial variables to convert inequaltiy to equality.

e.g. `A * x <= b` iff `A * x + s = b, s >= 0`.

#### Basic Solution
Let `A = [B D]` where `B` is a basis for column space, if `xb = b / B` and `xb >= 0`, then the basic solution `xb` is feasible. 
And the basic solution is actually `x = [xb 0]`.

Let `r = cb \ B`, where `c = [cb cd]`, if `r * D <= cd` then `xb` is dual feasible.

If `xb` is both primal and dual feasible, then it is optimum.

#### Two Phrase Method
Solve the artificial problem for a basic solution
```
min  1 * s
s.t. A * x + s = b,
     x, s >= 0
```
If the optimum value is zero, then the problem is feasible.


## Algorithm 

#### simplex.py 
1. revised simplex 
1. dual simplex 

#### decomposition.py
1. Dantzig Wolfe Decomposition

#### branch_cut.py
1. branch and bound
1. Gomory cut

### LU
#### pf_update.py
1. product form update

#### factorization.py
1. FT-like update

### Graph

#### dynamic.py
1. 0-1 knapsack problem

#### network_simplex.py
1. minimum cost flow problem 

#### transport.py
1. transportation problem

### Modeling

#### sudoku.py
1. Sudoku as 0-1 linear programming


## Dependency
numpy, scipy.


## Reference

### Books
1. Bertsekas D. P. "Network Optimization: Continuous and Discrete Models". 1998.
1. Luenberger D. G., Ye Yinyu. "Linear and Nonlinear Programming". 2008.
2. Pan Pingqi. "Linear Programming Computation". 2014. (中文版于2012年出版)
3. 刘红英, 夏勇等. "数学规划基础". 2012.

### Papers
1. Huangfu Q , Hall J A J . Novel update techniques for the revised simplex method[J]. Computational Optimization & Applications, 2015, 60(3):587-608.

### Further
1. H.D. Mittelmann, Decision Tree for Optimization Software, http://plato.asu.edu/guide.html.
1. Coin-OR https://www.coin-or.org/
1. cvxopt http://cvxopt.org/
1. MIPLIB http://miplib.zib.de/
1. MiniZinc http://www.minizinc.org/
1. NEOS https://neos-server.org/neos/
1. SCIP http://scip.zib.de/
1. LEMON https://lemon.cs.elte.hu/trac/lemon

