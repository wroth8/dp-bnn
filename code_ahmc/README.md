# Adaptive HMC

The code in this subdirectory is from the paper

```
[1] Ziyu Wang, Shakir Mohamed, and Nando de Freitas;
    Adaptive Hamiltonian and Riemann Manifold Monte Carlo; ICML 2013; pp. 1462-1470
```

The original code can be found at https://bitbucket.org/ziyuw/ahmc/src/master/

We made some slight modifications to the code as we had some issues with NaNs in our experiments.
In addition to the original `ahmc.m` file, we added two more versions.
The three versions differ in the way how HMC is initialized, i.e.,

* `ahmc.m`: Initial sample is obtained by minimizing the negative log-density, and minimization is initialized with a random vector
* `ahmcWithStart.m`: Initial sample is provided as an additional argument
* `ahmcWithStartOpt.m`: Initial sample is obtained by minimizing the negative log-density, and minimization is initialized with a vector provided as an additional argument
