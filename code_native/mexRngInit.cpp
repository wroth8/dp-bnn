// Initializes the C++ random number generator.
//
// Syntax:
// mexRngInit(uint32/str)
//
// Example:
// mexRngInit(uint32(12345))
// mexRngInit(mexRngState())
//
// @author Wolfgang Roth

#include <mex.h>

#include "utils_random.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int is_error = 0;
  unsigned int seed;

  if (nrhs != 1)
  {
    mexPrintf("Error in 'mexRngInit': Syntax is 'mexRngInit(rng_init)'\n");
    goto error;
  }
  
  if (nlhs != 0)
  {
    mexPrintf("Error in 'mexRngInit': Function does not have any output arguments\n");
    goto error;
  }
  
  // Check argument 'rng_init'
  if (mxIsChar(prhs[0]))
  {
    if (mxGetM(prhs[0]) != 1)
    {
      mexPrintf("Error in 'mexRngInit': Argument 'rng_init' must be either a state string or an uint32\n");
      goto error;
    }
    
    char* state_str = mxArrayToString(prhs[0]);
    if (state_str == NULL)
      goto memory_error;
    rnd_init_state(state_str, mxGetN(prhs[0]));
    mxFree(state_str);
  }
  else if (mxIsUint32(prhs[0]))
  {
    if (mxGetNumberOfElements(prhs[0]) != 1)
    {
      mexPrintf("Error in 'mexRngInit': Argument 'rng_init' must be either a state string or an uint32\n");
      goto error;
    }
    rnd_init_seed( *((unsigned int*) mxGetData(prhs[0])) );
  }
  else
  {
    mexPrintf("Error in 'mexRngInit': Argument 'rng_init' must be either a state string or an uint32\n");
    goto error;
  }
  
  goto cleanup;

memory_error:
  mexPrintf("Error in 'mexRngInit': Memory allocation failed\n");
  
error:
  is_error = 1;

cleanup:
  if (is_error)
    mexErrMsgTxt("Stopping exuction...\n");
}
