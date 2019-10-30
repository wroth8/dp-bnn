# Matlab Code

The main object is a neural network struct which is documented in the `dpnnInit`/`nnInit` functions.
Basically, there are two different versions of most functions:
(i) Functions for neural networks without weight sharing start with `nn` and (ii) functions for neural networks with weight sharing start with `dpnn`.

Use functions starting with `fun` as a wrapper for functions that require function/gradient handles operating on vectors rather than structs (e.g., `ahmc.m` in subdirectory `code_ahmc`).
