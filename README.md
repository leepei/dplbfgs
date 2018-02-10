# DPLBFGS - A Distributed Proximal LBFGS Method for Regularized Optimization

This code implements the algorithm used in the experiment of the following paper in C/C++ and MPI:
	_Lee, Ching-pei, Lim, Cong Han, Wright, Stephen J.. [Improving Communication Complexity of Distributed Non-smooth Regularized Minimization Using Previous Updates].

In additional to our algorithm, the following algorithms are also implemented.
- OWLQN
- SPARSA

## Getting started
To compile the code, you will need to install g++ and an implementation of MPI.
You will need to list the machines being used in a separate file, and make sure they are directly accessible through ssh.
Additionally the code depends on the BLAS and LAPACK libraries.

The code split.py, borrowed from [MPI-LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/distributed-liblinear/), partition the data and distribted the segments to the designated machines.
Then the program ./train solves the optimization problem to obtain a model.
