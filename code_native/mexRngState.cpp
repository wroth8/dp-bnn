// Returns the current state of the C++ random number generator.
//
// Syntax:
// rng_state = mexRngState()
//
// Return:
// The current state of the C++ random number generator as a char array
//
// @author Wolfgang Roth

#include <mex.h>

#include "utils_random.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int is_error = 0;
  int i;

  mwSize result_size[2];
  mxChar* result_char = NULL;

  char* state_str = NULL;
  int state_str_len = 0;

  state_str = rnd_state(&state_str_len);
  if (state_str == NULL)
    goto memory_error;

  result_size[0] = 1;
  result_size[1] = state_str_len;
  plhs[0] = mxCreateCharArray(2, result_size);
  if (plhs[0] == NULL)
    goto memory_error;
  
  result_char = (mxChar*) mxGetData(plhs[0]);
  for (i = 0; i < state_str_len; i++)
    result_char[i] = state_str[i];
  
  goto cleanup;
  
memory_error:
  mexPrintf("Error in 'mexRngState': Memory allocation failed\n");
  
error:
  is_error = 1;

cleanup:
  if (state_str != NULL)
    free(state_str);
    
  if (is_error)
    mexErrMsgTxt("Stopping exuction...\n");
}
