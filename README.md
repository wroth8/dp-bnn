# Bayesian Neural Networks with Weight Sharing Using Dirichlet Processes

This repository contains Matlab and C/C++ code used for the experiments in our paper

```
@ARTICLE{Roth2018,
    AUTHOR = {Wolfgang Roth and Franz Pernkopf},
    TITLE = {Bayesian Neural Networks with Weight Sharing Using {D}irichlet Processes}, 
    JOURNAL = {IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    YEAR = {2018}
}
```

## Usage

1. Clone this repository: `git clone https://github.com/wroth8/dp-bnn.git`
2. Compile mex-files as described in the `code_native` subdirectory
3. Run examples `example_*` or `experiment_housing.m` using **Matlab2013b**. The examples are intended to demonstrate how the code works and to illustrate the algorithms especially on low-dimensional data. Unfortunately, we cannot guarantee that the code is working on any other Matlab version since it was developed entirely on Matlab2013b and the behavior of used functions might have changed since then.

#### Adaptive HMC Code

With kind permission from Ziyu Wang, we uploaded the code from [1] that we used in this work to `code_ahmc`.

```
[1] Ziyu Wang, Shakir Mohamed, and Nando de Freitas;
    Adaptive Hamiltonian and Riemann Manifold Monte Carlo; ICML 2013; pp. 1462-1470
```
