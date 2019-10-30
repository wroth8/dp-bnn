// Sample weight indicators Z according to Neal's Algorithm 8 with likelihood
// interpolation.
//
// Syntax:
// mexSampleZInterpolate(model,x,t,batch_size,approx_N,m[,verbosity,approx_method])
//
// Arguments:
// model: A DPNN model struct for which the weight indicators should be
//   resampled. See documentation of dpnnInitCRP for its structure.
// x: NxD matrix of input features where N is the number of samples and Din is
//   the number of input features.
// t: Target values. For multiclass classification, this value should be a Nx1
//   column vector containing the class values. For other tasks, this value
//   should be a NxDout value where Dout is the dimension of the output.
// batch_size: The batch size used for the sampling algorithm. Use zero to not
//   use the mini-batch mode. In our preliminary tests, the mini-batch mode did
//   not yield good results.
// approx_N: The number of points used for likelihood interpolation, i.e., the
//   parameter 's' in the paper. Use 0 to disable likelihood interpolation. Note
//   that disabling likelihood interpolation can result in very high runtime.
// m: The number of auxiliary variables used in Neal's Algorithm 8.
// verbosity: Determines what is printed to the output:
//    0: No output
//    1: Output sampling process at layer level
//    2: Output sampling process at neuron level
//    3: Output sampling process at connection level
//    4: Full output (output will be quite cluttered)
// approx_method: The interpolation method to use. We refer to the documentation
//   of the matlab interpolation functions as they perform the same computations.
//   The following values are possible.
//    'none': No likelihood interpolation
//    'nearest': Nearest neighbor interpolation
//    'linear': Linear interpolation
//    'pchip': Cubic interpolation.
//
// Return:
// A DPNN model struct which resampled weight indicators.
//
// @author Wolfgang Roth

#include <mex.h>
#include <string.h>
#include "definitions.h"
#include "sampleZ.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int is_error = 0;
  int i = 0, j = 0, k = 0;
  
  const int num_struct_fields = 14;
  const char* struct_name_layout               = "layout";
  const char* struct_name_task                 = "task";
  const char* struct_name_sharing              = "sharing";
  const char* struct_name_activation           = "activation";
  const char* struct_name_num_layers           = "num_layers";
  const char* struct_name_ZW                   = "ZW";
  const char* struct_name_Zb                   = "Zb";
  const char* struct_name_W                    = "W";
  const char* struct_name_num_weights          = "num_weights";
  const char* struct_name_num_unique_weights   = "num_unique_weights";
  const char* struct_name_alpha                = "alpha";
  const char* struct_name_beta                 = "beta";
  const char* struct_name_gamma                = "gamma";
  const char* struct_name_rng_init_seed_matlab = "rng_init_seed_matlab";
  const char* struct_name_rng_init_seed_native = "rng_init_seed_native";
  const char* struct_names[] = { struct_name_layout, struct_name_task,
      struct_name_sharing, struct_name_activation, struct_name_num_layers,
      struct_name_ZW, struct_name_Zb, struct_name_W, struct_name_num_weights,
      struct_name_num_unique_weights, struct_name_alpha, struct_name_beta,
      struct_name_gamma, struct_name_rng_init_seed_matlab,
      struct_name_rng_init_seed_native };
  
  mxArray* mx_model_layout = NULL;
  mxArray* mx_model_task = NULL;
  mxArray* mx_model_sharing = NULL;
  mxArray* mx_model_activation = NULL;
  mxArray* mx_model_num_layers = NULL;
  mxArray* mx_model_ZW = NULL;
  mxArray* mx_model_Zb = NULL;
  mxArray* mx_model_W = NULL;
  mxArray* mx_model_num_weights = NULL;
  mxArray* mx_model_num_unique_weights = NULL;
  mxArray* mx_model_alpha = NULL;
  mxArray* mx_model_beta = NULL;
  mxArray* mx_model_gamma = NULL;
  mxArray* mx_model_rng_init_seed_matlab = NULL;
  mxArray* mx_model_rng_init_seed_native = NULL;
  
  mxArray* result_model = NULL;
  
  int* layout = NULL;
  int layout_len = 0;

  int task = -1;
  char* str_task = NULL;
  int str_task_len = 0;
  
  int sharing = -1;
  char* str_sharing = NULL;
  int str_sharing_len = 0;
  
  int activation = -1;
  char* str_activation = NULL;
  int str_activation_len = 0;
  
  int num_layers = -1;
  int** num_weights = NULL;
  int* num_unique_weights = NULL;
  
  int** ZW = NULL;
  int** Zb = NULL;
  double** W = NULL;
  
  double* alpha = NULL;
  double beta = -1;
  double* gamma = NULL;
  
  double* x = NULL;
  double* t = NULL;
  int N = -1; // Number of data samples
  int batch_size = 0; // batch size for sampleZ
  int approx_N   = 0; // approximation granularity for sigmoid/tanh activation
  int m          = 0; // number of new weights tried when resampling Z values
  int verbosity  = 0; // output level of sampleZ
  
  int approx_method = -1; // approximation (interpolation) method
  char* str_approx_method = NULL;
  int str_approx_method_len = 0;

  if (nrhs != 6 && nrhs != 7 && nrhs != 8)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Syntax is 'mexSampleZInterpolate(model,x,t,batch_size,approx_N,m[,verbosity,approx_method])'\n");
    goto error;
  }
  
  if (nlhs != 1)
  {
    mexPrintf("Warning in 'mexSampleZInterpolate': No assignment of return value\n");
  }
  
  result_model = mxDuplicateArray(prhs[0]);
  if (result_model == NULL)
    goto memory_error;

  // Check argument 'model'
  if (!mxIsStruct(prhs[0]))
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Argument 'model' must be a struct\n");
    goto error;
  }
  mx_model_layout = mxGetField(result_model, 0, struct_name_layout);
  mx_model_task = mxGetField(result_model, 0, struct_name_task);
  mx_model_sharing = mxGetField(result_model, 0, struct_name_sharing);
  mx_model_activation = mxGetField(result_model, 0, struct_name_activation);
  mx_model_num_layers = mxGetField(result_model, 0, struct_name_num_layers);
  mx_model_ZW = mxGetField(result_model, 0, struct_name_ZW);
  mx_model_Zb = mxGetField(result_model, 0, struct_name_Zb);
  mx_model_W = mxGetField(result_model, 0, struct_name_W);
  mx_model_num_weights = mxGetField(result_model, 0, struct_name_num_weights);
  mx_model_num_unique_weights = mxGetField(result_model, 0, struct_name_num_unique_weights);
  mx_model_alpha = mxGetField(result_model, 0, struct_name_alpha);
  mx_model_beta = mxGetField(result_model, 0, struct_name_beta);
  mx_model_gamma = mxGetField(result_model, 0, struct_name_gamma);
  mx_model_rng_init_seed_matlab = mxGetField(result_model, 0, struct_name_rng_init_seed_matlab);
  mx_model_rng_init_seed_native = mxGetField(result_model, 0, struct_name_rng_init_seed_native);
  
  // Check if all necessary values are present in 'model'
  if (mx_model_layout == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'layout' missing in 'model'\n");
    goto error;
  }
  if (mx_model_task == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'task' missing in 'model'\n");
    goto error;
  }
  if (mx_model_sharing == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'sharing' missing in 'model'\n");
    goto error;
  }
  if (mx_model_activation == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'activation' missing in 'model'\n");
    goto error;
  }
  if (mx_model_num_layers == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_layers' missing in 'model'\n");
    goto error;
  }
  if (mx_model_ZW == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'ZW' missing in 'model'\n");
    goto error;
  }
  if (mx_model_Zb == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'Zb' missing in 'model'\n");
    goto error;
  }
  if (mx_model_W == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'W' missing in 'model'\n");
    goto error;
  }
  if (mx_model_num_unique_weights == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_unique_weights' missing in 'model'\n");
    goto error;
  }
  if (mx_model_alpha == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'alpha' missing in 'model'\n");
    goto error;
  }
  if (mx_model_beta == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'beta' missing in 'model'\n");
    goto error;
  }
  if (mx_model_gamma == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'gamma' missing in 'model'\n");
    goto error;
  }
  if (mx_model_rng_init_seed_matlab == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'rng_init_seed_matlab' missing in 'model'\n");
    goto error;
  }
  if (mx_model_rng_init_seed_native == NULL)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'rng_init_seed_native' missing in 'model'\n");
    goto error;
  }
  
  //-----------------------------------------------------------------------
  // Read 'layout'
  if (!mxIsInt32(mx_model_layout) || mxGetM(mx_model_layout) != 1)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'layout' in 'model' must be an int32 row vector\n");
    goto error;
  }
  layout_len = mxGetN(mx_model_layout);
  layout = (int*) mxGetData(mx_model_layout);
  
  //-----------------------------------------------------------------------
  // Read 'task'
  if (!mxIsChar(mx_model_task))
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'task' in 'model' must be a string\n");
    goto error;
  }
  str_task_len = mxGetN(mx_model_task);
  str_task = (char*) malloc(sizeof(char) * (str_task_len + 1));
  if (str_task == NULL)
    goto memory_error;
  mxGetString(mx_model_task, str_task, str_task_len + 1);
  if (!strcmp(str_task, "regress"))
  {
    task = TASK_REGRESSION;
  }
  else if (!strcmp(str_task, "biclass"))
  {
    task = TASK_BINARY_CLASSIFICATION;
  }
  else if (!strcmp(str_task, "muclass"))
  {
    task = TASK_MULTICLASS_CLASSIFICATION;
  }
  else
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'task' in 'model' must be one of 'regress', 'biclass' or 'muclass'\n");
    goto error;
  }

  //-----------------------------------------------------------------------
  // Read 'sharing'
  if (!mxIsChar(mx_model_sharing))
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'sharing' in 'model' must be a string\n");
    goto error;
  }
  str_sharing_len = mxGetN(mx_model_sharing);
  str_sharing = (char*) malloc(sizeof(char) * (str_sharing_len + 1));
  if (str_sharing == NULL)
    goto memory_error;
  mxGetString(mx_model_sharing, str_sharing, str_sharing_len + 1);
  if (!strcmp(str_sharing, "layerwise"))
  {
    sharing = SHARING_LAYERWISE;
  }
  else if (!strcmp(str_sharing, "global"))
  {
    sharing = SHARING_GLOBAL;
  }
  else
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'sharing' in 'model' must be one of 'layerwise' or 'global'\n");
    goto error;
  }
  
  //-----------------------------------------------------------------------
  // Read 'activation'
  if (!mxIsChar(mx_model_activation))
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'activation' in 'model' must be a string\n");
    goto error;
  }
  str_activation_len = mxGetN(mx_model_activation);
  str_activation = (char*) malloc(sizeof(char) * (str_activation_len + 1));
  if (str_activation == NULL)
    goto memory_error;
  mxGetString(mx_model_activation, str_activation, str_activation_len + 1);
  if (!strcmp(str_activation, "sigmoid"))
  {
    activation = ACTIVATION_SIGMOID;
  }
  else if (!strcmp(str_activation, "tanh"))
  {
    activation = ACTIVATION_TANH;
  }
  else if (!strcmp(str_activation, "relu"))
  {
    activation = ACTIVATION_RELU;
  }
  else
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'activation' in 'model' must be one of 'sigmoid', 'tanh' or 'relu'\n");
    goto error;
  }
  
  //-----------------------------------------------------------------------
  // Read 'num_layers'
  if (!mxIsInt32(mx_model_num_layers) || mxGetNumberOfElements(mx_model_num_layers) != 1)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_layers' in 'model' must be a int32 scalar\n");
    goto error;
  }
  num_layers = (int) *((int*)mxGetData(mx_model_num_layers));
  if (num_layers != layout_len - 1)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_layers' in 'model' contains inconsistent data\n");
    goto error;
  }
  
  //-----------------------------------------------------------------------
  // Read 'num_unique_weights'
  if (sharing == SHARING_LAYERWISE)
  {
    if (!mxIsInt32(mx_model_num_unique_weights) || mxGetM(mx_model_num_unique_weights) != 1 || mxGetN(mx_model_num_unique_weights) != num_layers)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_unique_weights' in 'model' must be an int32 row vector with 'num_layers' elements\n");
      goto error;
    }
    num_unique_weights = (int*) mxGetData(mx_model_num_unique_weights);
  }
  else // SHARING_GLOBAL
  {
    if (!mxIsInt32(mx_model_num_unique_weights) || mxGetNumberOfElements(mx_model_num_unique_weights) != 1)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_unique_weights' in 'model' must be an int32\n");
      goto error;
    }
    num_unique_weights = (int*) mxGetData(mx_model_num_unique_weights);
  }
  
  //-----------------------------------------------------------------------
  // Read 'num_weights'
  num_weights = (int**) calloc(num_layers, sizeof(int*));
  if (num_weights == NULL)
    goto memory_error;
  if (sharing == SHARING_LAYERWISE)
  {
    if (!mxIsCell(mx_model_num_weights) || mxGetM(mx_model_num_weights) != 1 || mxGetN(mx_model_num_weights) != num_layers)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_weights' in 'model' must be a cell array with 'num_layers' elements\n");
      goto error;
    }
    for (i = 0; i < num_layers; i++)
    {
      mxArray* cell = mxGetCell(mx_model_num_weights, i);
      if (!mxIsInt32(cell) || mxGetM(cell) != 1 || mxGetN(cell) != num_unique_weights[i])
      {
        mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_weights' in 'model' contains inconsistent data\n");
        goto error;
      }
      num_weights[i] = (int*) mxGetData(cell);
    }
  }
  else // SHARING_GLOBAL
  {
    if (!mxIsInt32(mx_model_num_weights) || mxGetM(mx_model_num_weights) != 1 || mxGetN(mx_model_num_weights) != *num_unique_weights)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'num_weights' in 'model' must be an int32 row vector with num_unique_weights elements\n");
      goto error;
    }
    for (i = 0; i < num_layers; i++)
    {
      num_weights[i] = (int*) mxGetData(mx_model_num_weights);
    }
  }
  
  //-----------------------------------------------------------------------
  // Read 'ZW'
  if (!mxIsCell(mx_model_ZW) || mxGetM(mx_model_ZW) != 1 || mxGetN(mx_model_ZW) != num_layers)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'ZW' in 'model' must be a cell array with 'num_layers' elements\n");
    goto error;
  }
  ZW = (int**) calloc(num_layers, sizeof(int*));
  if (ZW == NULL)
    goto memory_error;
  for (i = 0; i < num_layers; i++)
  {
    mxArray* cell = mxGetCell(mx_model_ZW, i);
    if (!mxIsInt32(cell) || mxGetM(cell) != layout[i] || mxGetN(cell) != layout[i+1])
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'ZW' in 'model' contains inconsistent data\n");
      goto error;
    }
    ZW[i] = (int*) mxGetData(cell);
  }
  
  //-----------------------------------------------------------------------
  // Read 'ZW'
  if (!mxIsCell(mx_model_Zb) || mxGetM(mx_model_Zb) != 1 || mxGetN(mx_model_Zb) != num_layers)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'Zb' in 'model' must be a cell array with 'num_layers' elements\n");
    goto error;
  }
  Zb = (int**) calloc(num_layers, sizeof(int*));
  if (Zb == NULL)
    goto memory_error;
  for (i = 0; i < num_layers; i++)
  {
    mxArray* cell = mxGetCell(mx_model_Zb, i);
    if (!mxIsInt32(cell) || mxGetM(cell) != 1 || mxGetN(cell) != layout[i+1])
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'Zb' in 'model' contains inconsistent data\n");
      goto error;
    }
    Zb[i] = (int*) mxGetData(cell);
  }
  
  //-----------------------------------------------------------------------
  // Read 'W'
  W = (double**) calloc(num_layers, sizeof(double*));
  if (W == NULL)
    goto memory_error;
  if (sharing == SHARING_LAYERWISE)
  {
    if (!mxIsCell(mx_model_W) || mxGetM(mx_model_W) != 1 || mxGetN(mx_model_W) != num_layers)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'W' in 'model' must be a cell array with 'num_layers' elements\n");
      goto error;
    }
    for (i = 0; i < num_layers; i++)
    {
      mxArray* cell = mxGetCell(mx_model_W, i);
      if (!mxIsNumeric(cell) || mxGetM(cell) != 1 || mxGetN(cell) != num_unique_weights[i])
      {
        mexPrintf("Error in 'mexSampleZInterpolate': Field 'W' in 'model' contains inconsistent data\n");
        goto error;
      }
      W[i] = mxGetPr(cell);
    }
  }
  else // SHARING_GLOBAL
  {
    if (!mxIsNumeric(mx_model_W) && mxGetM(mx_model_W) != 1 && mxGetN(mx_model_W) != num_unique_weights[0])
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'W' in 'model' must be a row vector with 'num_unique_weights' elements\n");
      goto error;
    }
    for (i = 0; i < num_layers; i++)
    {
      W[i] = mxGetPr(mx_model_W);
    }
  }

  //-----------------------------------------------------------------------
  // Read 'alpha'
  if (sharing == SHARING_LAYERWISE)
  {
    if (!mxIsNumeric(mx_model_alpha) || mxGetM(mx_model_alpha) != 1 || mxGetN(mx_model_alpha) != num_layers)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'alpha' in 'model' must be row vector with 'num_layers' numeric elements\n");
      goto error;
    }
    alpha = mxGetPr(mx_model_alpha);
    for (i = 0; i < num_layers; i++)
    {
      if (alpha[i] <= 0)
      {
        mexPrintf("Error in 'mexSampleZInterpolate': Field 'alpha' in 'model' must contain only positive entries\n");
        goto error;
      }
    }
  }
  else // SHARING_GLOBAL
  {
    if (!mxIsNumeric(mx_model_alpha) || mxGetNumberOfElements(mx_model_alpha) != 1)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'alpha' in 'model' must be scalar\n");
      goto error;
    }
    alpha = mxGetPr(mx_model_alpha);
    if (*alpha <= 0)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'alpha' in 'model' must be positive\n");
    }
  }
  
  //-----------------------------------------------------------------------
  // Read 'beta'
  if (!mxIsNumeric(mx_model_beta) || mxGetNumberOfElements(mx_model_beta) != 1)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'beta' in 'model' must be numeric scalar\n");
  }
  beta = *mxGetPr(mx_model_beta);
  if (task == TASK_REGRESSION && beta <= 0)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Field 'beta' in 'model' must be postive\n");
  }

  //-----------------------------------------------------------------------
  // Read 'gamma'
  if (sharing == SHARING_LAYERWISE)
  {
    if (!mxIsNumeric(mx_model_gamma) || mxGetM(mx_model_gamma) != 1 || mxGetN(mx_model_gamma) != num_layers)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'gamma' in 'model' must be row vector with 'num_layers' numeric elements\n");
      goto error;
    }
    gamma = mxGetPr(mx_model_gamma);
    for (i = 0; i < num_layers; i++)
    {
      if (gamma[i] <= 0)
      {
        mexPrintf("Error in 'mexSampleZInterpolate': Field 'gamma' in 'model' must contain only positive entries\n");
        goto error;
      }
    }
  }
  else // SHARING_GLOBAL
  {
    if (!mxIsNumeric(mx_model_gamma) || mxGetNumberOfElements(mx_model_gamma) != 1)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'gamma' in 'model' must be scalar\n");
      goto error;
    }
    gamma = mxGetPr(mx_model_gamma);
    if (*gamma <= 0)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Field 'gamma' in 'model' must be positive\n");
    }
  }
  
  //-----------------------------------------------------------------------
  // Read 'x'
  if (!mxIsNumeric(prhs[1]) || mxGetN(prhs[1]) != layout[0])
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Argument 'x' must be a matrix with 'layout(1)' columns\n");
    goto error;
  }
  x = (double*) mxGetPr(prhs[1]);
  N = mxGetM(prhs[1]);
  
  //-----------------------------------------------------------------------
  // Read 't'
  if (task != TASK_MULTICLASS_CLASSIFICATION)
  {
    if (!mxIsNumeric(prhs[2]) || mxGetM(prhs[2]) != N || mxGetN(prhs[2]) != layout[num_layers])
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Argument 't' must be a matrix with 'size(x,1)' rows and 'layout(end)' columns\n");
      goto error;
    }
  }
  else
  {
    if (!mxIsNumeric(prhs[2]) || mxGetM(prhs[2]) != N || mxGetN(prhs[2]) != 1)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Argument 't' must be a column vector of length 'size(x,1)'\n");
      goto error;
    }
  }
  t = (double*) mxGetPr(prhs[2]);
  
  //-----------------------------------------------------------------------
  // Read 'batch_size'
  if (!mxIsNumeric(prhs[3]) || mxGetNumberOfElements(prhs[3]) != 1)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Argument 'batch_size' must be a scalar\n");
    goto error;
  }
  batch_size = (int) *mxGetPr(prhs[3]);

  //-----------------------------------------------------------------------
  // Read 'approx_N'
  if (!mxIsNumeric(prhs[4]) || mxGetNumberOfElements(prhs[4]) != 1)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Argument 'approx_N' must be a scalar\n");
    goto error;
  }
  approx_N = (int) *mxGetPr(prhs[4]);
  
  //-----------------------------------------------------------------------
  // Read 'm'
  if (!mxIsNumeric(prhs[5]) || mxGetNumberOfElements(prhs[5]) != 1)
  {
    mexPrintf("Error in 'mexSampleZInterpolate': Argument 'm' must be a scalar\n");
    goto error;
  }
  m = (int) *mxGetPr(prhs[5]);
  
  //-----------------------------------------------------------------------
  // Read 'verbosity'
  if (nrhs >= 7)
  {
    if (!mxIsNumeric(prhs[6]) || mxGetNumberOfElements(prhs[6]) != 1)
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Argument 'verbosity' must be a scalar\n");
      goto error;
    }
    verbosity = (int) *mxGetPr(prhs[6]);
  }
  else
  {
    verbosity = VERBOSITY_NONE;
  }
  
  //-----------------------------------------------------------------------
  // Read 'approx_method'
  if (nrhs >= 8)
  {
    if (!mxIsChar(prhs[7]))
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Argument 'approx_method' must be a string\n");
      goto error;
    }
    str_approx_method_len = mxGetN(prhs[7]);
    str_approx_method = (char*) malloc(sizeof(char) * (str_approx_method_len + 1));
    if (str_approx_method == NULL)
      goto memory_error;
    mxGetString(prhs[7], str_approx_method, str_approx_method_len + 1);
    if (!strcmp(str_approx_method, "none"))
    {
      approx_method = INTERPOLATE_NONE;
    }
    else if (!strcmp(str_approx_method, "nearest"))
    {
      approx_method = INTERPOLATE_NEAREST;
    }
    else if (!strcmp(str_approx_method, "linear"))
    {
      approx_method = INTERPOLATE_LINEAR;
    }
    else if (!strcmp(str_approx_method, "pchip"))
    {
      approx_method = INTERPOLATE_PCHIP;
    }
    else
    {
      mexPrintf("Error in 'mexSampleZInterpolate': Argument 'approx_method' must be one of 'none', 'nearest' (default), 'linear' or 'pchip'\n");
      goto error;
    }
  }
  else
  {
    approx_method = INTERPOLATE_NEAREST;
  }
  
  mexPrintf("mexSampleZInterpolate: N=%d, batch_size=%d, approx_N=%d, m=%d, verbosity=%d", N, batch_size, approx_N, m, verbosity);
  switch (approx_method)
  {
  case INTERPOLATE_NONE: mexPrintf(", approx_method=NONE\n"); break;
  case INTERPOLATE_NEAREST: mexPrintf(", approx_method=NEAREST\n"); break;
  case INTERPOLATE_LINEAR: mexPrintf(", approx_method=LINEAR\n"); break;
  case INTERPOLATE_PCHIP: mexPrintf(", approx_method=PCHIP\n"); break;
  default: mexPrintf(", approx_method=UNKNOWN (we should not be here\n"); break;
  }

  is_error = sampleZ_NealAlgorithm8(layout, task, sharing, activation,
    num_layers, ZW, Zb, &W, &num_weights, num_unique_weights, alpha, beta,
    gamma, x, t, N, batch_size, approx_N, approx_method, m, verbosity);
  if (is_error)
    goto memory_error;

  // Resize weights of the result model
  if (sharing == SHARING_LAYERWISE)
  {
    for (i = 0; i < num_layers; i++)
    {
      mxArray* cell_W = NULL;
      mxArray* cell_num_weights = NULL;
      
      cell_W = mxGetCell(mx_model_W, i);
      mxSetPr(cell_W, W[i]);
      mxSetM(cell_W, 1);
      mxSetN(cell_W, num_unique_weights[i]);
      
      cell_num_weights = mxGetCell(mx_model_num_weights, i);
      mxSetData(cell_num_weights, num_weights[i]);
      mxSetM(cell_num_weights, 1);
      mxSetN(cell_num_weights, num_unique_weights[i]);
    }
  }
  else // SHARING_GLOBAL
  {
    mxSetPr(mx_model_W, *W);
    mxSetM(mx_model_W, 1);
    mxSetN(mx_model_W, *num_unique_weights);

    mxSetData(mx_model_num_weights, *num_weights);
    mxSetM(mx_model_num_weights, 1);
    mxSetN(mx_model_num_weights, *num_unique_weights);
  }

  plhs[0] = result_model;

  // Everything went fine, skip error section and go to cleanup section
  goto cleanup;

memory_error:
  mexPrintf("Error in 'mexSampleZInterpolate': Memory allocation failed\n");

error:
  is_error = 1;
  
  mxDestroyArray(result_model);
  plhs[0] = NULL;

cleanup:
  if (str_task != NULL)
    free(str_task);

  if (str_sharing != NULL)
    free(str_sharing);

  if (str_activation != NULL)
    free(str_activation);

  if (str_approx_method != NULL)
    free(str_approx_method);

  if (num_weights != NULL)
    free(num_weights);

  if (ZW != NULL)
    free(ZW);

  if (Zb != NULL)
    free(Zb);

  if (W != NULL)
    free(W);

  if (is_error)
    mexErrMsgTxt("Stopping exuction...\n");
}
