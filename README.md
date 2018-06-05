# DPLBFGS - A Distributed Proximal LBFGS Method for Regularized Optimization

This code implements the algorithm used in the experiment of the following paper in C/C++ and MPI:
_Lee, Ching-pei, Lim, Cong Han, Wright, Stephen J. [A Distributed Quasi-Newton Algorithm for Empirical Risk
	Minimization with Nonsmooth Regularization](http://www.optimization-online.org/DB_HTML/2018/03/6500.html). The 24th ACM SIGKDD
	International Conference on Knowledge Discovery and Data Mining, 2018._

In additional to our algorithm, the following algorithms are also implemented.
- OWLQN
- SPARSA

## Getting started
To compile the code, you will need to install g++ and an implementation of MPI.
You will need to list the machines being used in a separate file, and make sure they are directly accessible through ssh.
Additionally the code depends on the BLAS and LAPACK libraries.

The code split.py, borrowed from [MPI-LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/), partition the data and distribted the segments to the designated machines.
Then the program ./train solves the optimization problem to obtain a model.

## Problem being solved

The code solves the L1-regularized logistic regression problem.

min_{w} |w|_1 + C \sum_{i=1}^n \log(1 + \exp(- y_i w^T x_i))

with a user-specified parameter C > 0.
