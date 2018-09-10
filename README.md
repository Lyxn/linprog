# Linear Programming 

### Introduction
Linear programing problem in standard form
```
min  c * x
s.t. A * x = b,
     x >= 0.
```
The dual problem is 
```
max r * b
s.t. r * A <= c
```
In the integer programming, the variables must be integers.

#### Artificial Variable
One can use artificial variables to convert inequaltiy to equality.

e.g. `A * x <= b` iff `A * x + s = b, s>=0`.

#### Basic Solution
Let `A = [B D]` where `B` is not singular, if `xb = b / B` and `xb >= 0`, then the basic solution `xb` is feasible.

Let `r = cb \ B`, where `c = [cb cd]`, if `rb * D <= cd` then `xb` is dual feasible.

If `xb` is both primal and dual feasible, then it is optimum.

#### Two Pharse Method
Solve the artificial problem for a basic solution
```
min  1 * s
s.t. A * x + s = b,
     x, s >= 0
```
if the minimum value is zero, then the problem is feasible.


### Algorithm 

#### simplex.py 
1. revised simplex 
1. dual simplex 

#### decomposition.py
1. Dantzig Wolfe Decomposition

#### dynamic.py
1. 0-1 knapsack problem


### Dependency
Numpy, Scipy.


### Reference
1. Luenberger D. G., Ye Yinyu. "Linear and Nonlinear Programming".


### Further
1. Coin-OR https://www.coin-or.org/
1. cvxopt http://cvxopt.org/
1. MIPLIB http://miplib.zib.de/
1. MiniZinc http://www.minizinc.org/

