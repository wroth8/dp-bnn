//
//
// @author Wolfgang Roth

#ifndef __UTILS_RANDOM_H__
#define __UTILS_RANDOM_H__

// Initializes the random number generator with the given seed and sets the next
// standard normal distributed random number to 'not available'.
extern "C" void rnd_init_seed(unsigned int seed);

// Initializes the random number generator with the given state.
extern "C" void rnd_init_state(const char* state_str, int state_str_length);

// Returns the current state of the random number generator.
extern "C" char* rnd_state(int* state_str_length);

// Returns a uniformly distributed random number between zero and one.
extern "C" double rnd_unif();

// Returns a standard normal distributed random number.
extern "C" double rnd_norm();

// Computes k distinct values in the range {0,...,N-1} and stores them in
// permutation. Permutation must be of size N, but only the first k elements
// of permutation contain valid data.
extern "C" void rnd_permutation(int N, int k, int* permutation);

#endif


