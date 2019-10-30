//
//
// @author Wolfgang Roth

#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <random>
#include <sstream>

std::mt19937 rnd_engine;
std::uniform_real_distribution<double> rnd_distr_unif(0.0,1.0);
std::normal_distribution<double> rnd_distr_norm(0.0,1.0);

extern "C" void rnd_init_seed(unsigned int seed)
{
  rnd_engine.seed(seed);
}

extern "C" void rnd_init_state(const char* state_str, int state_str_length)
{
  std::stringstream stream(std::string(state_str, state_str_length));
  stream >> rnd_engine;
}

extern "C" char* rnd_state(int* state_str_length)
{
  char* state_str = NULL;
  std::stringstream stream;
  stream << rnd_engine;
  state_str = (char*) malloc(sizeof(char) * (stream.str().length() + 1));
  if (state_str == NULL)
  {
    *state_str_length = 0;
    return NULL;
  }
  // Copy state content
  for (unsigned int i = 0; i < stream.str().length(); i++)
    state_str[i] = stream.str().c_str()[i];
  state_str[stream.str().length()] = '\0';
  *state_str_length = stream.str().length();
  return state_str;
}

extern "C" double rnd_unif()
{
  // Note:
  // Reseting the state here does not seem to have any impact as opposed to the
  // normal distribution (see below).
  double val = rnd_distr_unif(rnd_engine);
  rnd_distr_unif.reset();
  return val;
}

extern "C" double rnd_norm()
{
  // Note:
  // The normal_distribution implementation seems to generate two normal random
  // variables each time (see Box-Muller method). This has the effect that if
  // the internal state of the engine is changed using rnd_init_seed or
  // rnd_init_state, the second variable is kept in the distribution memory and
  // returned on the next call to rnd_distr_norm although it was generated
  // using the old state. Therefore we reset the distribution each time and
  // effectively throw away half the random numbers which should not be a
  // problem since generating random numbers should not be the bottleneck.
  double val = rnd_distr_norm(rnd_engine);
  rnd_distr_norm.reset();
  return val;
}

// TODO: Implement it according to the description in the header file (only k
// elements are needed)
extern "C" void rnd_permutation(int N, int k, int* permutation)
{
  for (int i = 0; i < N; i++)
    permutation[i] = i;
  for (int i = N - 1; i > 0; i--)
  {
    std::uniform_int_distribution<int> rnd_distr_int(0, i);
    std::swap(permutation[i], permutation[rnd_distr_int(rnd_engine)]);
  }
}
