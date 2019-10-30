# C/C++ Code

Due to efficiency reasons, we implemented the code for configuration sampling (z-sampling) in C.
We obtained the mex-files using the following procedure.

**In a terminal:**
Compile `utils_random` to a shared object file using

```
g++ -O3 -c -std=c++11 -Wall -Werror -fpic utils_random.cpp -o utils_random.o
g++ -shared -o libutils_random.so utils_random.o
```

**In Matlab:**
Compile the mex-files using:

```
mex LDFLAGS='\$LDFLAGS -Wl,-rpath=\$\ORIGIN' mexRngState.cpp libutils_random.so
mex LDFLAGS='\$LDFLAGS -Wl,-rpath=\$\ORIGIN' mexRngInit.cpp libutils_random.so
mex LDFLAGS='\$LDFLAGS -Wl,-rpath=\$\ORIGIN' mexSampleZInterpolate.cpp sampleZ.cpp utils.cpp libutils_random.so -lcblas
```

### Note about GCC Version
In our experiments, we used `Matlab2013b` and `GCC 4.2.7`.
For `libutils_random.so` it seems to be important that it is compiled with `GCC 4.2.7` because we get runtime errors in Matlab when we use the newer version `GCC 6.3.0`.
