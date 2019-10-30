// Implementation of the functions defined in sampleZ.h (see there for more
// details)
//
// @author Wolfgang Roth

#include "sampleZ.h"
#include "utils.h"
#include "utils_random.h"
#include "definitions.h"

#include <stdlib.h>
#include <math.h>
#include <cblas.h>

double beta_inv_0_5;
double approx_N_0_5;
double* buf_sum = NULL;
double* buf_max = NULL;

#define APPROXIMATE_ACTIVATION

#define VAL_2_POW_M4 0.0625
#define VAL_2_POW_M3 0.125
#define VAL_2_POW_3 8.0
#define VAL_2_POW_4 16.0

// todo: check if all parameters of all functions are actually used
// todo: restrict keyword

int sampleZ_NealAlgorithm8(int* layout, int task, int sharing,
    int activation, int num_layers, int** ZW, int** Zb, double*** W_ptr,
    int*** num_weights_ptr, int *num_unique_weights, double* alpha, double beta,
    double* gamma, double* x, double* t_total, int N_total, int N_batch,
    int approx_N, int approx_method, int m, int verbosity)
{
  int error = 0;
  int i, j, k;
  int l;
  int n;

  void* aux_ptr = NULL; // auxiliary pointer for realloc calls

  int batch_mode = (N_batch > 0 && N_batch < N_total);
  int* batch_permutation = NULL;
  int N = (batch_mode ? N_batch : N_total);

  double likelihood_last_layer = 0;

  double** W = *W_ptr;
  int** num_weights = *num_weights_ptr;

  double** W_full = NULL;
  double** b_full = NULL;

  double** a = NULL;
  double** z = NULL;
  double* t = NULL;

  double likelihood_max;
  double* approx_likelihoods = NULL;

  double* approx_x = NULL;
  double* approx_h_inv = NULL;
  double* pchip_delta1 = NULL;
  double* pchip_delta2 = NULL;
  double* pchip_b = NULL;
  double* pchip_c = NULL;
  double* pchip_d = NULL;

  int* approx_idx_relu_lookup = NULL;

  double sqrt_gamma = -1;
  double log_alpha_div_m = -1;

  int empty_indices_size = 0;
  
  int memsize_W;
  int memsize_num_weights;

  double* log_num_weights = NULL;
  int memsize_log_num_weights = 0;

  double* likelihoods = NULL;
  int memsize_likelihoods = 0;

  int* empty_indices = NULL;
  int memsize_empty_indices;

  double* softmax_logsumexp = NULL;

  // debugging stuff, can be removed afterwards
  double** debug_a = NULL;
  double** debug_z = NULL;

#ifdef APPROXIMATE_ACTIVATION
  if (activation == ACTIVATION_SIGMOID)
    my_sigmoid_init();
  else if (activation == ACTIVATION_TANH)
    my_tanh_init();

  if (task == TASK_BINARY_CLASSIFICATION)
    softplus_approx_init();
#endif

  debug_a = (double**) calloc(num_layers + 1, sizeof(double*));
  if (debug_a == NULL)
    goto memory_error;

  debug_z = (double**) calloc(num_layers + 1, sizeof(double*));
  if (debug_z == NULL)
    goto memory_error;

  debug_z[0] = x; // todo: adjust to batch_mode?

  for (l = 1; l < num_layers + 1; l++)
  {
    debug_a[l] = (double*) malloc(sizeof(double) * N * layout[l]);
    if (debug_a[l] == NULL)
      goto memory_error;

    debug_z[l] = (double*) malloc(sizeof(double) * N * layout[l]);
    if (debug_z[l] == NULL)
      goto memory_error;
  }


  beta_inv_0_5 = 0.5 / beta;
  approx_N_0_5 = 0.5 * (double) approx_N;

  W_full = (double**) calloc(num_layers, sizeof(double*));
  if (W_full == NULL)
    goto memory_error;
  for (l = 0; l < num_layers; l++)
  {
    W_full[l] = (double*) malloc(sizeof(double) * layout[l] * layout[l + 1]);
    if (W_full[l] == NULL)
      goto memory_error;

    for (j = 0; j < layout[l + 1]; j++)
    {
      for (i = 0; i < layout[l]; i++)
      {
        W_full[l][i + layout[l] * j] = W[l][ZW[l][i + layout[l] * j] - 1];
      }
    }
  }

  b_full = (double**) calloc(num_layers, sizeof(double*));
  if (b_full == NULL)
    goto memory_error;
  for (l = 0; l < num_layers; l++)
  {
    b_full[l] = (double*) malloc(sizeof(double) * layout[l + 1]);
    if (b_full[l] == NULL)
      goto memory_error;

    for (i = 0; i < layout[l + 1]; i++)
      b_full[l][i] = W[l][Zb[l][i] - 1];
  }

  memsize_empty_indices = 256;
  empty_indices = (int*) malloc(sizeof(int) * memsize_empty_indices);
  if (empty_indices == NULL)
    goto memory_error;

  if (approx_N == 0 && approx_method != INTERPOLATE_NONE)
  {
    _printf("Warning: Interpolation will be disabled. (approx_N == 0 and approx_method != NONE)\n");
    approx_method = INTERPOLATE_NONE;
  }
  if (approx_N > 0 && approx_method == INTERPOLATE_NONE)
  {
    _printf("Warning: Interpolation will be disabled. (approx_N > 0 and approx_method == NONE)\n");
    approx_N = 0;
  }

  if (approx_N > 0)
  {
    approx_likelihoods = (double*) malloc(sizeof(double) * N * (approx_N + 1));
    if (approx_likelihoods == NULL)
      goto memory_error;

    approx_x = (double*) malloc(sizeof(double) * (approx_N + 1));
    if (approx_x == NULL)
      goto memory_error;

    for (i = 0; i < approx_N + 1; i++)
    {
      switch (activation)
      {
      case ACTIVATION_SIGMOID:
        approx_x[i] = (double) i / (double) approx_N;
        break;
      case ACTIVATION_TANH:
        approx_x[i] = 2.0 * (double) i / (double) approx_N - 1.0;
        break;
      case ACTIVATION_RELU:
        if (i == 0)
          approx_x[i] = 0.0;
        else
          approx_x[i] = pow(2.0, -4.0 + (double) i); // starts at 2^-3 since i=1
        break;
      default:
        _printf("Error in 'sampleZ.c': Unknown activation function\n");
        break;
      }
    }

    if (approx_method == INTERPOLATE_PCHIP)
    {
      pchip_delta1 = (double*) malloc(sizeof(double) * N);
      if (pchip_delta1 == NULL)
        goto memory_error;

      pchip_delta2 = (double*) malloc(sizeof(double) * N);
      if (pchip_delta2 == NULL)
        goto memory_error;

      // Polynomial value b for intervals
      pchip_b = (double*) malloc(sizeof(double) * N * approx_N);
      if (pchip_b == NULL)
        goto memory_error;

      // Polynomial value c for intervals
      pchip_c = (double*) malloc(sizeof(double) * N * approx_N);
      if (pchip_c == NULL)
        goto memory_error;

      // Slopes at each x-value == polynomial value d for intervals
      // Note that pchip_d is larger than pchip_b and pchip_c. The slope
      // of the last-value is only needed to precompute the b and c values
      // of the last interval and it is not needed for interpolation.
      pchip_d = (double*) malloc(sizeof(double) * N * (approx_N + 1));
      if (pchip_d == NULL)
        goto memory_error;
    }

    if (activation == ACTIVATION_RELU)
    {
      int lookup_size = -1;
      if (approx_method == INTERPOLATE_NEAREST)
        lookup_size = (int) (approx_x[approx_N] / VAL_2_POW_M4) + 1;
      else if (approx_method == INTERPOLATE_LINEAR || approx_method == INTERPOLATE_PCHIP)
        lookup_size = (int) (approx_x[approx_N - 1] / VAL_2_POW_M3) + 1;
      else
        _printf("Error in sampleZ: Unknown interpolation (we should not be here)\n");

      approx_idx_relu_lookup = (int*) malloc(sizeof(int) * lookup_size);
      if (approx_idx_relu_lookup == NULL)
        goto memory_error;

      initReluIdxLookup(approx_idx_relu_lookup, approx_method, approx_N + 1, approx_x);

      if (approx_method == INTERPOLATE_LINEAR)
      {
        approx_h_inv = (double*) malloc(sizeof(double) * approx_N);
        if (approx_h_inv == NULL)
          goto memory_error;
        for (i = 0; i < approx_N; i++)
          approx_h_inv[i] = 1.0 / (approx_x[i+1] - approx_x[i]);
      }
    }
  }

  a = (double**) calloc(num_layers + 1, sizeof(double*));
  if (a == NULL)
    goto memory_error;

  z = (double**) calloc(num_layers, sizeof(double*)); // Note: Output activations are never computed
  if (z == NULL)
    goto memory_error;

  if (batch_mode)
  {
    z[0] = (double*) malloc(sizeof(double) * N * layout[0]);
    if (z[0] == NULL)
      goto memory_error;

    t = (double*) malloc(sizeof(double) * N * (task == TASK_MULTICLASS_CLASSIFICATION ? 1 : layout[num_layers]));
    if (t == NULL)
      goto memory_error;

    batch_permutation = (int*) malloc(sizeof(double) * N_total);
    if (batch_permutation == NULL)
      goto memory_error;
  }
  else
  {
    z[0] = x;
    t = t_total;
  }

  for (l = 1; l < num_layers + 1; l++)
  {
    a[l] = (double*) malloc(sizeof(double) * N * layout[l]);
    if (a[l] == NULL)
      goto memory_error;
  }

  for (l = 1; l < num_layers; l++)
  {
    z[l] = (double*) malloc(sizeof(double) * N * layout[l]);
    if (z[l] == NULL)
      goto memory_error;
  }

  if (task == TASK_MULTICLASS_CLASSIFICATION)
  {
    buf_max = (double*) malloc(sizeof(double) * N);
    if (buf_max == NULL)
      goto memory_error;

    buf_sum = (double*) malloc(sizeof(double) * N);
    if (buf_sum == NULL)
      goto memory_error;

    softmax_logsumexp = (double*) malloc(sizeof(double) * N);
    if (softmax_logsumexp == NULL)
      goto memory_error;
  }

  if (!batch_mode)
  {
    feedforwardLayer(W_full[0], b_full[0], z[0], a[1], layout[0], layout[1], N);
    if (num_layers > 1)
      activationFunction(a[1], z[1], layout[1], N, activation);
  }

  for (l = 0; l < num_layers; l++)
  {
    _printf_verbosity(verbosity, VERBOSITY_LAYER, "Sampling layer %d/%d\n", l + 1, num_layers);
    int l2;
    int last_layer = (l == num_layers - 1);
    int second_to_last_layer = (l == num_layers - 2);
    int unique_weights = (sharing == SHARING_LAYERWISE ? num_unique_weights[l] : *num_unique_weights);

    // Allocate some memory in powers of two
    memsize_W = get_next_power2(unique_weights + m);
    memsize_num_weights = memsize_W;

    aux_ptr = _realloc(W[l], sizeof(double) * memsize_W);
    if (aux_ptr == NULL)
      goto memory_error;
    else if (sharing == SHARING_LAYERWISE)
      W[l] = (double*) aux_ptr;
    else
      for (l2 = 0; l2 < num_layers; l2++)
        W[l2] = (double*) aux_ptr;

    aux_ptr = _realloc(num_weights[l], sizeof(int) * memsize_num_weights);
    if (aux_ptr == NULL)
      goto memory_error;
    else if (sharing == SHARING_LAYERWISE)
      num_weights[l] = (int*) aux_ptr;
    else
      for (l2 = 0; l2 < num_layers; l2++)
        num_weights[l2] = (int*) aux_ptr;

    if (memsize_W > memsize_log_num_weights)
    {
      memsize_log_num_weights = memsize_W;
      aux_ptr = realloc(log_num_weights, sizeof(double) * memsize_log_num_weights);
      if (aux_ptr == NULL)
        goto memory_error;
      else
        log_num_weights = (double*) aux_ptr;
    }

    if (memsize_W > memsize_likelihoods)
    {
      memsize_likelihoods = memsize_W;
      aux_ptr = realloc(likelihoods, sizeof(double) * memsize_likelihoods);
      if (aux_ptr == NULL)
        goto memory_error;
      else
        likelihoods = (double*) aux_ptr;
    }

    //if (l <= 1)
    //  continue;

    // Precompute logs of num_weights and other frequently used values
    for (k = 0; k < unique_weights; k++)
      log_num_weights[k] = log(num_weights[l][k]);
    log_alpha_div_m = log((sharing == SHARING_LAYERWISE ? alpha[l] : *alpha) / m);
    sqrt_gamma = sqrt(sharing == SHARING_LAYERWISE ? gamma[l] : *gamma);

    // Reset the queue for empty indices
    empty_indices_size = 0;

    if (!batch_mode)
    {
      if (!last_layer)
      {
        // There is at least one more layer to sample from after the current.
        // No need to compute activations since they will be immediately
        // invalidated with feedforwardNeuronNegative in the next step.
        feedforwardLayer(W_full[l + 1], b_full[l + 1], z[l + 1], a[l + 2], layout[l + 1], layout[l + 2], N);
      }
      else
      {
        // We are in the last layer. The activation function is now given by the
        // task. likelihood_last_layer stores the current error. By updating only a
        // single edge we can update the error incrementally and don't need to
        // compute the full output layer in case of regression or binary
        // classification.
        if (task == TASK_REGRESSION || task == TASK_BINARY_CLASSIFICATION)
          likelihood_last_layer = computeErrors(t, a[l + 1], layout[l + 1], N, task);
      }
    }

    for (i = 0; i < layout[l + 1]; i++)
    {
      _printf_verbosity(verbosity, VERBOSITY_NEURON, "Sampling neuron %d/%d  [#unique_weights=%d]\n", i, layout[l + 1], unique_weights - empty_indices_size);
      //------------------------------------------------------------------------
      // Sample all weight indicators of edges feeding into neuron 'i'
      //------------------------------------------------------------------------

      if (batch_mode) // add something like "&& i % 10 == 0" to reuse batches for more iterations to save computation
      {
        // Select a random sample of N elements
        rnd_permutation(N_total, N, batch_permutation);
        for (j = 0; j < layout[0]; j++)
        {
          for (n = 0; n < N; n++)
          {
            z[0][n + N * j] = x[batch_permutation[n] + N_total * j];
          }
        }
        for (j = 0; j < (task == TASK_MULTICLASS_CLASSIFICATION ? 1 : layout[num_layers]); j++)
        {
          for (n = 0; n < N; n++)
          {
            t[n + N * j] = t_total[batch_permutation[n] + N_total * j];
          }
        }

        // Feedforward up to the current layer
        feedforwardLayer(W_full[0], b_full[0], z[0], a[1], layout[0], layout[1], N);
        if (num_layers > 1)
          activationFunction(a[1], z[1], layout[1], N, activation);

        for (l2 = 0; l2 <= l; l2++)
        {
          int last_layer = (l2 == num_layers - 1);

          if (!last_layer)
          {
            feedforwardLayer(W_full[l2 + 1], b_full[l2 + 1], z[l2 + 1], a[l2 + 2], layout[l2 + 1], layout[l2 + 2], N);
            if (l2 != l && l2 != num_layers - 2)
              activationFunction(a[l2 + 2], z[l2 + 2], layout[l2 + 2], N, activation);
          }
          else
          {
            if (task == TASK_REGRESSION || task == TASK_BINARY_CLASSIFICATION)
              likelihood_last_layer = computeErrors(t, a[l2 + 1], layout[l2 + 1], N, task);
          }
        }
      }

      if (!last_layer)
      {
        // Remove the influence of neuron 'i' on the following layer.
        feedforwardNeuronNegative(W_full[l + 1], z[l + 1], a[l + 2], layout[l + 1], layout[l + 2], N, i);

        if (approx_N > 0)
        {
          // Approximate the changes on the neuron with the discretization trick
          computeApproximation(&W_full[l + 1], &b_full[l + 1], t, &a[l + 1],
              &z[l + 1], &layout[l + 1], num_layers - l - 1, N, activation, task,
              i, approx_likelihoods, approx_N, approx_x);
          if (approx_method == INTERPOLATE_PCHIP)
            computePchipCoefficients(pchip_delta1, pchip_delta2, pchip_b, pchip_c, pchip_d,
                approx_likelihoods, approx_x, N, approx_N + 1);
        }
      }
      else
      {
        if (task == TASK_REGRESSION || task == TASK_BINARY_CLASSIFICATION)
          // Remove the error from the current output 'i'
          likelihood_last_layer -= computeErrorsNeuron(t, a[l + 1], layout[l + 1], N, task, i);
        else
          // Precompute logsumexp for all output neurons except output neuron 'i'
          initActivationFunctionSoftmax(a[l + 1], layout[l + 1], N, softmax_logsumexp, i);
      }

      // TODO: We can probably save one computation in each loop since it must have been computed in the last iteration (i guess this is only a minor gain)
      for (j = 0; j < layout[l] + 1; j++)
      {
        _printf_verbosity(verbosity, VERBOSITY_EDGE, "Sampling edge (%d,%d) [#unique_weights=%d]\n", i, j, unique_weights - empty_indices_size);
        //----------------------------------------------------------------------
        // Sample ZW_{i,j}; ZW_{i,end}=Zb_{i}
        //----------------------------------------------------------------------
        int bias = (j == layout[l]);
        int k_old;
        int singleton_weight;

        if (!bias)
          k_old = ZW[l][j + layout[l] * i];
        else
          k_old = Zb[l][i];

        // If the current weight is a singleton weight, it is added (implicitly) as one of the m new weights.
        // In this case the weight gets a special treatment (see case distinctions in the following code).
        singleton_weight = (num_weights[l][k_old - 1] == 1);

        num_weights[l][k_old - 1]--;
        log_num_weights[k_old - 1] = log(num_weights[l][k_old - 1]);

        // Remove influence of current Z_{i,j}
        if (!bias)
          updateConnectionNegative(W_full[l], z[l], a[l + 1], layout[l], N, j, i);
        else
          updateBiasNegative(b_full[l], a[l + 1], N, i);

        // Sample m new values and append them to the weight vector (it is guaranteed that enough memory is allocated)
        for (n = 0; n < (singleton_weight ? m-1 : m); n++)
          W[l][unique_weights + n] = rnd_norm() * sqrt_gamma;

        // Plug in all values for the weights and compute the likelihoods
        likelihood_max = -INFINITY;
        for (k = 1; k <= unique_weights + (singleton_weight ? m-1 : m); k++) // Note: k follows the Matlab indices as in ZW/Zb
        {
          // Skip the weight if it is not used, i.e. it is currently in empty_indices
          if (isnan(W[l][k - 1]))
            continue;

          if (!bias)
          {
            ZW[l][j + layout[l] * i] = k;
            W_full[l][j + layout[l] * i] = W[l][k - 1]; // -1 for Matlab-C index differences
          }
          else
          {
            Zb[l][i] = k;
            b_full[l][i] = W[l][k - 1];
          }

          // Add influence of new weight
          if (!bias)
            updateConnection(W_full[l], z[l], a[l + 1], layout[l], N, j, i);
          else
            updateBias(b_full[l], a[l + 1], N, i);

          if (!last_layer)
          {
            activationFunctionSingle(a[l + 1], z[l + 1], layout[l + 1], N, activation, i);

            if (approx_N > 0)
            {
              // Compute approximated log-likelihoods
              double* z_tmp = z[l + 1] + N * i;
              likelihoods[k - 1] = 0;
              if (approx_method == INTERPOLATE_NEAREST)
              {
                for (n = 0; n < N; n++)
                {
                  int approx_idx = 0;
                  if (activation == ACTIVATION_SIGMOID)
                    approx_idx = (int) (z_tmp[n] * (double) approx_N + 0.5); // calling round() takes more time
                  else if (activation == ACTIVATION_TANH)
                    approx_idx = (int) ((z_tmp[n] + 1.0) * approx_N_0_5 + 0.5);
                  else if (activation == ACTIVATION_RELU)
                  {
                    if (z_tmp[n] <= 0)
                      approx_idx = 0;
                    else if (z_tmp[n] >= approx_x[approx_N])
                      approx_idx = approx_N;
                    else
                      approx_idx = approx_idx_relu_lookup[(int) (z_tmp[n] * VAL_2_POW_4)];
                  }
                  else
                    _printf("Error in 'sampleZ.c': Unknown activation function\n");

                  likelihoods[k - 1] += approx_likelihoods[n + N * approx_idx];
                }
              }
              else if (approx_method == INTERPOLATE_LINEAR)
              {
                double approx_aux = 0.0; // auxiliary variable
                double approx_t = 0.0; // t y0 + (1-t) y1
                int approx_idx = 0; // idx of y0
                for (n = 0; n < N; n++)
                {
                  if (activation == ACTIVATION_SIGMOID)
                  {
                    approx_aux = z_tmp[n] * (double) approx_N;
                    approx_idx = (int) approx_aux;
                    approx_idx = approx_idx < 0 ? 0 : approx_idx;
                    approx_idx = approx_idx > approx_N - 1 ? approx_N - 1 : approx_idx;
                    approx_t = approx_aux - (double) approx_idx;
                    likelihoods[k - 1] += ((1.0 - approx_t) * approx_likelihoods[n + N * approx_idx]);
                    likelihoods[k - 1] += (approx_t * approx_likelihoods[n + N * (approx_idx + 1)]);
                  }
                  else if (activation == ACTIVATION_TANH)
                  {
                    approx_aux = (z_tmp[n] + 1.0) * approx_N_0_5;
                    approx_idx = (int) approx_aux;
                    approx_idx = approx_idx < 0 ? 0 : approx_idx;
                    approx_idx = approx_idx > approx_N - 1 ? approx_N - 1 : approx_idx;
                    approx_t = approx_aux - (double) approx_idx;
                    likelihoods[k - 1] += ((1.0 - approx_t) * approx_likelihoods[n + N * approx_idx]);
                    likelihoods[k - 1] += (approx_t * approx_likelihoods[n + N * (approx_idx + 1)]);
                  }
                  else if (activation == ACTIVATION_RELU)
                  {
                    if (z_tmp[n] <= 0.0)
                    {
                      likelihoods[k - 1] += approx_likelihoods[n];
                    }
                    else
                    {
                      if (z_tmp[n] >= approx_x[approx_N - 1])
                        approx_idx = approx_N - 1;
                      else
                        approx_idx = approx_idx_relu_lookup[(int) (z_tmp[n] * VAL_2_POW_3)];
                      approx_t = (z_tmp[n] - approx_x[approx_idx]) * approx_h_inv[approx_idx];
                      likelihoods[k - 1] += ((1.0 - approx_t) * approx_likelihoods[n + N * approx_idx]);
                      likelihoods[k - 1] += (approx_t * approx_likelihoods[n + N * (approx_idx + 1)]);
                    }
                  }
                  else
                    _printf("Error in 'sampleZ.c': Unknown activation function\n");
                }
              }
              else if (approx_method == INTERPOLATE_PCHIP)
              {
                double approx_t = 0.0;
                int approx_idx = 0;
                int idx;
                for (n = 0; n < N; n++)
                {
                  if (activation == ACTIVATION_SIGMOID)
                  {
                    approx_idx = (int) (z_tmp[n] * (double) approx_N);
                    approx_idx = approx_idx < 0 ? 0 : approx_idx;
                    approx_idx = approx_idx > approx_N - 1 ? approx_N - 1 : approx_idx;
                    approx_t = z_tmp[n] - approx_x[approx_idx];
                    idx = n + N * approx_idx;
                    likelihoods[k - 1] += approx_likelihoods[idx] + approx_t * (pchip_d[idx] + approx_t * (pchip_c[idx] + approx_t * pchip_b[idx]));
                  }
                  else if (activation == ACTIVATION_TANH)
                  {
                    approx_idx = (int) ((z_tmp[n] + 1.0) * approx_N_0_5);
                    approx_idx = approx_idx < 0 ? 0 : approx_idx;
                    approx_idx = approx_idx > approx_N - 1 ? approx_N - 1 : approx_idx;
                    approx_t = z_tmp[n] - approx_x[approx_idx];
                    idx = n + N * approx_idx;
                    likelihoods[k - 1] += approx_likelihoods[idx] + approx_t * (pchip_d[idx] + approx_t * (pchip_c[idx] + approx_t * pchip_b[idx]));
                  }
                  else if (activation == ACTIVATION_RELU)
                  {
                    if (z_tmp[n] <= 0.0)
                    {
                      likelihoods[k - 1] += approx_likelihoods[n];
                    }
                    else
                    {
                      if (z_tmp[n] >= approx_x[approx_N - 1])
                        approx_idx = approx_N - 1;
                      else
                        approx_idx = approx_idx_relu_lookup[(int) (z_tmp[n] * VAL_2_POW_3)];
                      approx_t = z_tmp[n] - approx_x[approx_idx];
                      idx = n + N * approx_idx;
                      likelihoods[k - 1] += approx_likelihoods[idx] + approx_t * (pchip_d[idx] + approx_t * (pchip_c[idx] + approx_t * pchip_b[idx]));
                    }
                  }
                  else
                    _printf("Error in 'sampleZ.c': Unknown activation function\n");
                }
              }
              else
              {
                _printf("Error in 'sampleZ.c': Unknown approximation method\n");
              }
            }
            else
            {
              // Compute exact log-likelihoods

              // Feedforward to the next layer and compute the activation function if it is not the output layer
              feedforwardNeuron(W_full[l + 1], z[l + 1], a[l + 2], layout[l + 1], layout[l + 2], N, i);
              if (!second_to_last_layer)
                activationFunction(a[l + 2], z[l + 2], layout[l + 2], N, activation);

              // Feed forward up to the outputs
              feedforward(&W_full[l + 2], &b_full[l + 2], t, &a[l + 2], &z[l + 2], &layout[l + 2], num_layers - l - 2, N, activation, task);

              // Compute output activation function and log-likelihoods
              likelihoods[k - 1] = computeErrors(t, a[num_layers], layout[num_layers], N, task);

              // Remove influence on the next layer
              feedforwardNeuronNegative(W_full[l + 1], z[l + 1], a[l + 2], layout[l + 1], layout[l + 2], N, i);
            }
          }
          else
          {
            // We are in the last layer, compute output activation functions and log-likelihoods
            if (task == TASK_MULTICLASS_CLASSIFICATION)
              likelihoods[k - 1] = computeErrorsNeuronSoftmax(t, a[l + 1], N, softmax_logsumexp, i);
            else
              likelihoods[k - 1] = likelihood_last_layer + computeErrorsNeuron(t, a[l + 1], layout[l + 1], N, task, i);
          }

          if (k <= unique_weights && !(singleton_weight && k == k_old))
            likelihoods[k - 1] += log_num_weights[k - 1];
          else
            likelihoods[k - 1] += log_alpha_div_m;

          // Store maximum value for normalization stuff later (numerical stability)
          likelihood_max = fmax(likelihood_max, likelihoods[k - 1]);

          // Remove influence of new weight
          if (!bias)
            updateConnectionNegative(W_full[l], z[l], a[l + 1], layout[l], N, j, i);
          else
            updateBiasNegative(b_full[l], a[l + 1], N, i);
        }

        //----------------------------------------------------------------------
        // All log likelihoods are now computed, we can now sample the new
        // weight indicators proportional to the exponential of the
        // log-likelihoods
        //----------------------------------------------------------------------
        double exp_sum = 0;
        double random = rnd_unif();
        double cum_sum = 0;
        for (k = 0; k < unique_weights + (singleton_weight ? m-1 : m); k++)
        {
          if (isnan(W[l][k]))
            continue;
          likelihoods[k] = exp(likelihoods[k] - likelihood_max);
          exp_sum += likelihoods[k];
        }

        for (k = 0; k < unique_weights + (singleton_weight ? m-1 : m); k++)
        {
          if (isnan(W[l][k]))
            continue;
          cum_sum += likelihoods[k] / exp_sum;
          if (random <= cum_sum)
            break;
        }
        k++; // +1 for Matlab index differences

        //----------------------------------------------------------------------
        // Apply the changes to the structures. We can either simply replace
        // the weight indicators, remove weight indicators or add new weight
        // indicators.
        //----------------------------------------------------------------------
        if (k > unique_weights || (singleton_weight && k == k_old))
        {
          // We sampled a new weight (either old singleton weight or a newly
          // generated one).
          if (singleton_weight)
          {
            // Old weight was a singleton weight, we can put the newly sampled
            // weight to the position of the old singleton weight.
            if (!bias)
              ZW[l][j + layout[l] * i] = k_old;
            else
              Zb[l][i] = k_old;
            W[l][k_old - 1] = W[l][k - 1];
            num_weights[l][k_old - 1] = 1;
            log_num_weights[k_old - 1] = 0;
          }
          else
          {
            // Old weight was a non-singleton weight. First search if there are
            // some holes in the array were we could put the new weight.
            // Otherwise append it at the end.
            if (empty_indices_size > 0)
            {
              // We can reuse some old index stored in empty_indices
              empty_indices_size--;
              if (!bias)
                ZW[l][j + layout[l] * i] = empty_indices[empty_indices_size];
              else
                Zb[l][i] = empty_indices[empty_indices_size];
              W[l][empty_indices[empty_indices_size] - 1] = W[l][k - 1];
              num_weights[l][empty_indices[empty_indices_size] - 1] = 1;
              log_num_weights[empty_indices[empty_indices_size] - 1] = 0;
            }
            else
            {
              // We have to append the new weight at the end
              unique_weights++;
              if (unique_weights + m > memsize_W)
              {
                // Allocate another block which is twice as large as the current block
                memsize_W = get_next_power2(unique_weights + m);
                memsize_num_weights = memsize_W;

                aux_ptr = _realloc(W[l], memsize_W * sizeof(double));
                if (aux_ptr == NULL)
                  goto memory_error;
                else if (sharing == SHARING_LAYERWISE)
                  W[l] = (double*) aux_ptr;
                else
                  for (l2 = 0; l2 < num_layers; l2++)
                    W[l2] = (double*) aux_ptr;

                aux_ptr = _realloc(num_weights[l], memsize_num_weights * sizeof(int));
                if (aux_ptr == NULL)
                  goto memory_error;
                else if (sharing == SHARING_LAYERWISE)
                  num_weights[l] = (int*) aux_ptr;
                else
                  for (l2 = 0; l2 < num_layers; l2++)
                    num_weights[l2] = (int*) aux_ptr;
              }

              if (memsize_W > memsize_log_num_weights)
              {
                memsize_log_num_weights = memsize_W;
                aux_ptr = realloc(log_num_weights, memsize_log_num_weights * sizeof(double));
                if (aux_ptr == NULL)
                  goto memory_error;
                else
                  log_num_weights = (double*) aux_ptr;
              }

              if (memsize_W > memsize_likelihoods)
              {
                memsize_likelihoods = memsize_W;
                aux_ptr = realloc(likelihoods, memsize_likelihoods * sizeof(double));
                if (aux_ptr == NULL)
                  goto memory_error;
                else
                  likelihoods = (double*) aux_ptr;
              }

              if (!bias)
                ZW[l][j + layout[l] * i] = unique_weights;
              else
                Zb[l][i] = unique_weights;
              W[l][unique_weights - 1] = W[l][k - 1];
              num_weights[l][unique_weights - 1] = 1;
              log_num_weights[unique_weights - 1] = 0;
            }
          }
        }
        else
        {
          // We sampled some existing weight.
          if (singleton_weight)
          {
            // Old weight was a singleton weight. We have to invalidate the old
            // weight and queue its index up for future use (this creates a
            // hole)
            W[l][k_old - 1] = NAN;
            if (empty_indices_size + 1 > memsize_empty_indices)
            {
              // Allocate another block which is twice as large as the current block
              memsize_empty_indices = get_next_power2(empty_indices_size + 1);
              aux_ptr = realloc(empty_indices, memsize_empty_indices * sizeof(int));
              if (aux_ptr == NULL)
                goto memory_error;
              else
                empty_indices = (int*) aux_ptr;
            }
            empty_indices[empty_indices_size] = k_old;
            empty_indices_size++;

            if (!bias)
              ZW[l][j + layout[l] * i] = k;
            else
              Zb[l][i] = k;
            num_weights[l][k - 1]++;
            log_num_weights[k - 1] = log(num_weights[l][k - 1]);
          }
          else
          {
            // Old weight was a non-singleton weight. The old weight therefore
            // still stays valid and we just have to replace the indices.
            if (!bias)
              ZW[l][j + layout[l] * i] = k;
            else
              Zb[l][i] = k;
            num_weights[l][k - 1]++;
            log_num_weights[k - 1] = log(num_weights[l][k - 1]);
          }
        }

        if (!bias)
          W_full[l][j + layout[l] * i] = W[l][ZW[l][j + layout[l] * i] - 1];
        else
          b_full[l][i] = W[l][Zb[l][i] - 1];

        // We have finished the sampling of the current weight. We can now add
        // its influence. There is no need to call the activation function here.
        // In the next j-iteration its value will be immediately invalidated.
        if (!bias)
          updateConnection(W_full[l], z[l], a[l + 1], layout[l], N, j, i);
        else
          updateBias(b_full[l], a[l + 1], N, i);
      } // j-loop

      // We have finished sampling of all weights feeding into the current
      // neuron. We can now add its influence on the next layer.
      if (!last_layer)
      {
        activationFunctionSingle(a[l + 1], z[l + 1], layout[l + 1], N, activation, i);
        feedforwardNeuron(W_full[l + 1], z[l + 1], a[l + 2], layout[l + 1], layout[l + 2], N, i);
        if (!second_to_last_layer)
          activationFunction(a[l + 2], z[l + 2], layout[l + 2], N, activation);
      }
      else
      {
        // Add the error from the current output 'i'
        if (task == TASK_REGRESSION || task == TASK_BINARY_CLASSIFICATION)
          likelihood_last_layer += computeErrorsNeuron(t, a[l + 1], layout[l + 1], N, task, i);
      }

      //debug_layer_values(ZW, Zb, W, t, a, z, debug_a, debug_z, layout, num_layers, N, activation, task, last_layer ? l + 1 : l + 2);
    } // i-loop

    //--------------------------------------------------------------------------
    // The following loop fills holes in the arrays W[l] that are possibly
    // there due to resampling singleton weights. The code iterates from the
    // left of the array until a hole is found and replaces this value by a
    // non-hole value which was found by iterating from the right of the
    // array.
    //--------------------------------------------------------------------------
    i = 0;
    j = unique_weights;
    while (i < j)
    {
      // Invariants:
      // (1) there are no holes to the left of i
      // (2) j points onto a hole and there are only holes to the right of j
      for (; i < j; i++)
      {
        // Search for a hole
        if (isnan(W[l][i]))
          break;
      }
      if (i >= j)
      {
        // No hole found: j points onto a hole and there is no more hole to the left of j
        unique_weights = j;
        break;
      }
      for (j = j - 1; i < j; j--)
      {
        // Search for a non-hole
        if (!isnan(W[l][j]))
          break;
      }
      if (i >= j)
      {
        // No non-hole found: i points onto a hole and there are only non-holes to its right
        unique_weights = i;
        break;
      }
      W[l][i] = W[l][j];
      num_weights[l][i] = num_weights[l][j]; // no need to update log_num_weights: will be updated in next l-loop iteration anyway

      // Replace j in ZW/Zb with i
      int idx1, idx2;
      if (sharing == SHARING_LAYERWISE)
      {
        for (idx2 = 0; idx2 < layout[l + 1]; idx2++)
        {
          for (idx1 = 0; idx1 < layout[l]; idx1++)
          {
            if (ZW[l][idx1 + layout[l] * idx2] == j + 1)
              ZW[l][idx1 + layout[l] * idx2] = i + 1;
          }
        }
        for (idx1 = 0; idx1 < layout[l + 1]; idx1++)
        {
          if (Zb[l][idx1] == j + 1)
            Zb[l][idx1] = i + 1;
        }
      }
      else // sharing == SHARING_GLOBAL
      {
        for (l2 = 0; l2 < num_layers; l2++)
        {
          for (idx2 = 0; idx2 < layout[l2 + 1]; idx2++)
          {
            for (idx1 = 0; idx1 < layout[l2]; idx1++)
            {
              if (ZW[l2][idx1 + layout[l2] * idx2] == j + 1)
                ZW[l2][idx1 + layout[l2] * idx2] = i + 1;
            }
          }
          for (idx1 = 0; idx1 < layout[l2 + 1]; idx1++)
          {
            if (Zb[l2][idx1] == j + 1)
              Zb[l2][idx1] = i + 1;
          }
        }
      }
    }

    //--------------------------------------------------------------------------
    // Reallocate memory according to the total amount of weights that are now
    // used.
    //--------------------------------------------------------------------------
    aux_ptr = _realloc(W[l], sizeof(double) * unique_weights);
    if (aux_ptr == NULL)
      goto memory_error;
    else if (sharing == SHARING_LAYERWISE)
      W[l] = (double*) aux_ptr;
    else
      for (l2 = 0; l2 < num_layers; l2++)
        W[l2] = (double*) aux_ptr;

    aux_ptr = _realloc(num_weights[l], sizeof(int) * unique_weights);
    if (aux_ptr == NULL)
      goto memory_error;
    else if (sharing == SHARING_LAYERWISE)
      num_weights[l] = (int*) aux_ptr;
    else
      for (l2 = 0; l2 < num_layers; l2++)
        num_weights[l2] = (int*) aux_ptr;

    if (sharing == SHARING_LAYERWISE)
      num_unique_weights[l] = unique_weights;
    else
      *num_unique_weights = unique_weights;
  }

  goto cleanup;

memory_error:
  error = 1;
  _printf("A memory error has occurred in sampleZ.c\n");
    
cleanup:

  // Store possibly new memory locations due to realloc
  *W_ptr = W;
  *num_weights_ptr = num_weights;

  if (empty_indices != NULL)
    free(empty_indices);

  // debug stuff, can be removed afterwards
  if (debug_a != NULL)
  {
    for (i = 1; i < num_layers + 1; i++)
    {
      if (debug_a[i] != NULL)
        free(debug_a[i]);
    }
    free(debug_a);
  }
  if (debug_z != NULL)
  {
    for (i = 1; i < num_layers + 1; i++)
    {
      if (debug_z[i] != NULL)
        free(debug_z[i]);
    }
    free(debug_z);
  }
  //

  if (W_full != NULL)
  {
    for (i = 0; i < num_layers; i++)
    {
      if (W_full[i] != NULL)
        free(W_full[i]);
    }
    free(W_full);
  }

  if (b_full != NULL)
  {
    for (i = 0; i < num_layers; i++)
    {
      if (b_full[i] != NULL)
        free(b_full[i]);
    }
    free(b_full);
  }

  if (a != NULL)
  {
    for (i = 1; i < num_layers + 1; i++) // a[0] is always NULL
    {
      if (a[i] != NULL)
        free(a[i]);
    }
    free(a);
  }

  if (z != NULL)
  {
    if (batch_mode && z[0] != NULL)
      free(z[0]);
    // in case of no batch mode z[0] contains x and must not be freed
    // z[num_layers] is not used
    for (i = 1; i < num_layers; i++)
    {
      if (z[i] != NULL)
        free(z[i]);
    }
    free(z);
  }

  if (batch_mode && t != NULL)
    free(t);

  if (batch_permutation != NULL)
    free(batch_permutation);

  if (buf_max != NULL)
  {
    free(buf_max);
    buf_max = NULL;
  }

  if (buf_sum != NULL)
  {
    free(buf_sum);
    buf_sum = NULL;
  }

  if (softmax_logsumexp != NULL)
    free(softmax_logsumexp);

#ifdef ROW_CACHING
  if (buf_row != NULL)
  {
    free(buf_row);
    buf_row = NULL;
  }
#endif

  if (likelihoods != NULL)
    free(likelihoods);

  if (log_num_weights != NULL)
    free(log_num_weights);

  if (approx_likelihoods != NULL)
    free(approx_likelihoods);

  if (approx_x != NULL)
    free(approx_x);

  if (approx_h_inv != NULL)
    free(approx_h_inv);

  if (pchip_delta1 != NULL)
    free(pchip_delta1);

  if (pchip_delta2 != NULL)
    free(pchip_delta2);

  if (pchip_b != NULL)
    free(pchip_b);

  if (pchip_c != NULL)
    free(pchip_c);

  if (pchip_d != NULL)
    free(pchip_d);

  if (approx_idx_relu_lookup != NULL)
    free(approx_idx_relu_lookup);

  return error;
}

void feedforward(double** W, double** b, double* t, double** a,
    double** z, int* layout, int num_layers, int N, int activation, int task)
{
  int l;
  for (l = 0; l < num_layers; l++)
  {
    if (l != num_layers - 1)
    {
      feedforwardLayer(W[l], b[l], z[l], a[l + 1], layout[l], layout[l + 1], N);
      activationFunction(a[l + 1], z[l + 1], layout[l + 1], N, activation);
    }
    else
    {
      feedforwardLayer(W[l], b[l], z[l], a[l + 1], layout[l], layout[l + 1], N);
    }
  }
}

double computeErrors(double* t, double* a, int D_o, int N, int task)
{
  int i, n;
  double delta = 0;
  double error = 0;

  double* a_tmp = a;

  //------------------------------------------------------------------
  // Compute errors (log likelihood)
  if (task == TASK_REGRESSION)
  {
    for (i = 0; i < D_o; i++)
    {
      for (n = 0; n < N; n++)
      {
        delta = (a[n + N * i] - t[n + N * i]);
        error += delta * delta;
      }
    }
    error *= -beta_inv_0_5;
  }
  else if (task == TASK_BINARY_CLASSIFICATION)
  {
    for (i = 0; i < D_o; i++)
    {
      for (n = 0; n < N; n++)
      {
#ifdef APPROXIMATE_ACTIVATION
        if (t[n + N * i] == 1)
          error -= softplus_approx(-a[n + N * i]);
        else
          error -= softplus_approx(a[n + N * i]);
#else
        if (t[n + N * i] == 1)
          error -= softplus(-a[n + N * i]);
        else
          error -= softplus(a[n + N * i]);
#endif
      }
    }
  }
  else // TASK_MULTICLASS_CLASSIFICATION
  {
    // Find maximum for each n
    for (n = 0; n < N; n++)
      buf_max[n] = a[n];
    for (i = 1, a_tmp = a + N; i < D_o; i++, a_tmp += N)
    {
      for (n = 0; n < N; n++)
      {
        buf_max[n] = fmax(buf_max[n], a_tmp[n]);
      }
    }

    // Compute sum of exponential for each n
    for (n = 0; n < N; n++)
      buf_sum[n] = exp(a[n] - buf_max[n]);
    for (i = 1, a_tmp = a + N; i < D_o; i++, a_tmp += N)
    {
      for (n = 0; n < N; n++)
      {
        buf_sum[n] += exp(a_tmp[n] - buf_max[n]);
      }
    }

    // Compute error for the target class using the log-sum-exp trick
    for (n = 0; n < N; n++)
    {
      error = error + a[n + N * ((int) t[n] - 1)] - log(buf_sum[n]) - buf_max[n];
    }
  }
  return error;
}

double computeErrorsNeuron(double* t, double* a, int D_o, int N, int task,
    int neuron_index)
{
  int n;
  double delta = 0;
  double error = 0;
  double* a_tmp = a + N * neuron_index;
  double* t_tmp = t + N * neuron_index;

  //------------------------------------------------------------------
  // Compute errors (log likelihood)
  if (task == TASK_REGRESSION)
  {
    for (n = 0; n < N; n++)
    {
      delta = (a_tmp[n] - t_tmp[n]);
      error += delta * delta;
    }
    error *= -beta_inv_0_5;
  }
  else if (task == TASK_BINARY_CLASSIFICATION)
  {
    for (n = 0; n < N; n++)
    {
#ifdef APPROXIMATE_ACTIVATION
      if (t_tmp[n] == 1)
        error -= softplus_approx(-a_tmp[n]);
      else
        error -= softplus_approx(a_tmp[n]);
#else
      if (t_tmp[n] == 1)
        error -= softplus(-a_tmp[n]);
      else
        error -= softplus(a_tmp[n]);
#endif
    }
  }
  else // TASK_MULTICLASS_CLASSIFICATION
  {
    _printf("Error in sampleZ: We should not be here\n");
  }
  return error;
}

double computeErrorsNeuronSoftmax(double* t, double* a, int N,
    double* logsumexp, int output_index)
{
  int n;
  double* a_tmp = a + output_index * N;
  double logsumexp_inc;
  double logsumexp_inc_max;
  double error = 0;

  for (n = 0; n < N; n++)
  {
    error += a[n + ((int) t[n] - 1) * N];
    logsumexp_inc_max = fmax(logsumexp[n], a_tmp[n]);
#ifdef APPROXIMATE_ACTIVATION
    logsumexp_inc = softplus_negative_approx(-fabs(a_tmp[n] - logsumexp[n])) + logsumexp_inc_max;
#else
    //logsumexp_inc = log(exp(logsumexp[n] - logsumexp_inc_max) + exp(a_tmp[n] - logsumexp_inc_max)) + logsumexp_inc_max;
    logsumexp_inc = log(1.0 + exp(-fabs(a_tmp[n] - logsumexp[n]))) + logsumexp_inc_max;
#endif
    error -= logsumexp_inc;
  }
  return error;
}

void computeIndividualErrors(double* t, double* a, int D_o, int N, int task,
    double* errors)
{
  int i, n;
  double delta = 0;

  double* a_tmp;

  //------------------------------------------------------------------
  // Compute errors (log likelihood)
  if (task == TASK_REGRESSION)
  {
    for (n = 0; n < N; n++)
      errors[n] = 0;
    for (i = 0; i < D_o; i++)
    {
      for (n = 0; n < N; n++)
      {
        delta = (a[n + N * i] - t[n + N * i]);
        errors[n] += delta * delta;
      }
    }
    for (n = 0; n < N; n++)
      errors[n] *= -beta_inv_0_5;
  }
  else if (task == TASK_BINARY_CLASSIFICATION)
  {
    for (n = 0; n < N; n++)
      errors[n] = 0;
    for (i = 0; i < D_o; i++)
    {
      for (n = 0; n < N; n++)
      {
#ifdef APPROXIMATE_ACTIVATION
        if (t[n + N * i] == 1)
          errors[n] -= softplus_approx(-a[n + N * i]);
        else
          errors[n] -= softplus_approx(a[n + N * i]);
#else
        if (t[n + N * i] == 1)
          errors[n] -= softplus(-a[n + N * i]);
        else
          errors[n] -= softplus(a[n + N * i]);
#endif
      }
    }
  }
  else // TASK_MULTICLASS_CLASSIFICATION
  {
    // Find maximum for each n
    for (n = 0; n < N; n++)
      buf_max[n] = a[n];
    for (i = 1, a_tmp = a + N; i < D_o; i++, a_tmp += N)
    {
      for (n = 0; n < N; n++)
      {
        buf_max[n] = fmax(buf_max[n], a_tmp[n]);
      }
    }

    // Compute sum of exponential for each n
    for (n = 0; n < N; n++)
      buf_sum[n] = exp(a[n] - buf_max[n]);
    for (i = 1, a_tmp = a + N; i < D_o; i++, a_tmp += N)
    {
      for (n = 0; n < N; n++)
      {
        buf_sum[n] += exp(a_tmp[n] - buf_max[n]);
      }
    }

    // Compute error for the target class using the log-sum-exp trick
    for (n = 0; n < N; n++)
    {
      errors[n] = a[n + N * ((int) t[n] - 1)] - log(buf_sum[n]) - buf_max[n];
    }
  }
}

void feedforwardLayer(double* W, double* b, double* x, double* a, int D_x,
    int D_a, int N)
{
  int i, j;

  for (j = 0; j < D_a; j++)
    for (i = 0; i < N; i++)
      a[i + N * j] = b[j];

  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, D_a, D_x, 1, x, N, W, D_x, 1, a, N);
}

void feedforwardNeuron(double* W, double* x, double* a, int D_x, int D_a, int N,
    int neuron_index)
{
  int n, i;
  double* a_tmp = a;
  double* x_tmp = x + N * neuron_index;
  double w;

  for (i = 0; i < D_a; i++, a_tmp += N)
  {
    w = W[neuron_index + D_x * i];
    for (n = 0; n < N; n++)
    {
      a_tmp[n] += x_tmp[n] * w;
    }
  }
}

void feedforwardNeuronNegative(double* W, double* x, double* a, int D_x,
    int D_a, int N, int neuron_index)
{
  int n, i;
  double* a_tmp = a;
  double* x_tmp = x + N * neuron_index;
  double w;

  for (i = 0; i < D_a; i++, a_tmp += N)
  {
    w = W[neuron_index + D_x * i];
    for (n = 0; n < N; n++)
    {
      a_tmp[n] -= x_tmp[n] * w;
    }
  }
}

void updateConnection(double* W, double* x, double* a, int D_x, int N,
    int neuron_index_in, int neuron_index_out)
{
  int n;
  double* x_tmp = x + N * neuron_index_in;
  double* a_tmp = a + N * neuron_index_out;
  double w = W[neuron_index_in + D_x * neuron_index_out];

  for (n = 0; n < N; n++)
    a_tmp[n] += x_tmp[n] * w;
}

void updateConnectionNegative(double* W, double* x, double* a, int D_x, int N,
    int neuron_index_in, int neuron_index_out)
{
  int n;
  double* x_tmp = x + N * neuron_index_in;
  double* a_tmp = a + N * neuron_index_out;
  double w = W[neuron_index_in + D_x * neuron_index_out];

  for (n = 0; n < N; n++)
    a_tmp[n] -= x_tmp[n] * w;
}

void updateBias(double* b, double* a, int N, int neuron_index_out)
{
  int n;
  double* a_tmp = a + N * neuron_index_out;
  double w = b[neuron_index_out];

  for (n = 0; n < N; n++)
    a_tmp[n] += w;
}

void updateBiasNegative(double* b, double* a, int N, int neuron_index_out)
{
  int n;
  double* a_tmp = a + N * neuron_index_out;
  double w = b[neuron_index_out];

  for (n = 0; n < N; n++)
    a_tmp[n] -= w;
}

void activationFunction(double* a, double* z, int D_a, int N, int activation)
{
  // document: buf_max and buf_sum are only needed in case of softmax (i.e. multiclass classification)
  int n;
  int i;
  double* a_tmp = a;
  double* z_tmp = z;

  switch (activation)
  {
  case ACTIVATION_SIGMOID:
    for (i = 0; i < D_a; i++, z_tmp += N, a_tmp += N)
    {
      for (n = 0; n < N; n++)
      {
#ifdef APPROXIMATE_ACTIVATION
        z_tmp[n] = my_sigmoid(a_tmp[n]);
#else
        z_tmp[n] = 1 / (1 + exp(-a_tmp[n]));
#endif

      }
    }
    break;
  case ACTIVATION_TANH:
    for (i = 0; i < D_a; i++, z_tmp += N, a_tmp += N)
    {
      for (n = 0; n < N; n++)
      {
#ifdef APPROXIMATE_ACTIVATION
        z_tmp[n] = my_tanh(a_tmp[n]);
#else
        z_tmp[n] = tanh(a_tmp[n]);
#endif
      }
    }
    break;
  case ACTIVATION_RELU:
    for (i = 0; i < D_a; i++, z_tmp += N, a_tmp += N)
    {
      for (n = 0; n < N; n++)
      {
        z_tmp[n] = a_tmp[n] > 0 ? a_tmp[n] : 0;
      }
    }
    break;
  default:
    _printf("Error in 'sampleZ.c': Unknown activation function\n");
  }
}

void activationFunctionSingle(double* a, double* z, int D_a, int N,
    int activation, int neuron_index)
{
  // document: buf_max and buf_sum are only needed in case of softmax (i.e. multiclass classification)
  int n;
  double* a_tmp = NULL;
  double* z_tmp = NULL;

  switch (activation)
  {
  case ACTIVATION_SIGMOID:
    a_tmp = a + N * neuron_index;
    z_tmp = z + N * neuron_index;
    for (n = 0; n < N; n++)
    {
#ifdef APPROXIMATE_ACTIVATION
      z_tmp[n] = my_sigmoid(a_tmp[n]);
#else
      z_tmp[n] = 1 / (1 + exp(-a_tmp[n]));
#endif
    }
    break;
  case ACTIVATION_TANH:
    a_tmp = a + N * neuron_index;
    z_tmp = z + N * neuron_index;
    for (n = 0; n < N; n++)
    {
#ifdef APPROXIMATE_ACTIVATION
      z_tmp[n] = my_tanh(a_tmp[n]);
#else
      z_tmp[n] = tanh(a_tmp[n]);
#endif
    }
    break;
  case ACTIVATION_RELU:
    a_tmp = a + N * neuron_index;
    z_tmp = z + N * neuron_index;
    for (n = 0; n < N; n++)
    {
      z_tmp[n] = a_tmp[n] > 0 ? a_tmp[n] : 0;
    }
    break;
  default:
    _printf("Error in 'sampleZ.c': Unknown activation function\n");
  }
}

void initActivationFunctionSoftmax(double* a, int D_a, int N,
    double* logsumexp, int output_index)
{
  int i, n;
  int i0;
  double* a_tmp = NULL;

  // Compute maximum (leaving out neuron_index_out)
  i0 = output_index == 0 ? 1 : 0;
  a_tmp = a + i0 * N;
  for (n = 0; n < N; n++)
  {
    buf_max[n] = a_tmp[n];
  }

  for (i = i0 + 1, a_tmp = a + (i0 + 1) * N; i < D_a; i++, a_tmp += N)
  {
    if (i == output_index)
      continue;

    for (n = 0; n < N; n++)
    {
      buf_max[n] = fmax(buf_max[n], a_tmp[n]);
    }
  }

  // Compute log-sum-exp
  a_tmp = a + i0 * N;
  for (n = 0; n < N; n++)
  {
    logsumexp[n] = exp(a_tmp[n] - buf_max[n]);
  }
  for (i = i0 + 1, a_tmp = a + (i0 + 1) * N; i < D_a; i++, a_tmp += N)
  {
    if (i == output_index)
      continue;

    for (n = 0; n < N; n++)
    {
      logsumexp[n] += exp(a_tmp[n] - buf_max[n]);
    }
  }
  for (n = 0; n < N; n++)
  {
    logsumexp[n] = log(logsumexp[n]) + buf_max[n];
  }
}

void computeApproximation(double** W, double** b, double* t, double** a,
    double** z, int* layout, int num_layers, int N, int activation, int task,
    int approx_neuron, double* approx_likelihoods, int approx_N,
    double* approx_x)
{
  int n;
  int i;
  int last_layer = (num_layers == 1);
//  double approx_value = 0;

  for (i = 0; i < approx_N + 1; i++)
  {
    for (n = 0; n < N; n++)
      z[0][n + N * approx_neuron] = approx_x[i];

    // Feedforward to the next layer
    feedforwardNeuron(W[0], z[0], a[1], layout[0], layout[1], N, approx_neuron);

    // If we are not at the last layer, feed forward up to the outputs
    if (!last_layer)
    {
      activationFunction(a[1], z[1], layout[1], N, activation);
      feedforward(&W[1], &b[1], t, &a[1], &z[1], &layout[1], num_layers - 1, N, activation, task);
    }

    // Compute error function
    computeIndividualErrors(t, a[num_layers], layout[num_layers], N, task, &approx_likelihoods[N * i]);

    // Remove influence of the neuron on the next layer
    feedforwardNeuronNegative(W[0], z[0], a[1], layout[0], layout[1], N, approx_neuron);
  }
}

void computePchipCoefficients(double* pchip_delta1, double* pchip_delta2,
    double* pchip_b, double* pchip_c, double* pchip_d, double* lookup,
    double* x_vals, int lookup_N, int lookup_L)
{
  // Note: lookup_L is the number of x-values, i.e., approx_N + 1
  int i, n;
  double* lookup_ptr_l = NULL;
  double* lookup_ptr_m = NULL;
  double* lookup_ptr_r = NULL;
  double* pchip_b_ptr = NULL;
  double* pchip_c_ptr = NULL;
  double* pchip_d_ptr = NULL; // The slope at the given x values
  double* pchip_delta_swap = NULL;
  double h1, h2, h1_inv, h2_inv, h1h2_inv, h1sq_inv;
  double w1, w2, w1w2;

  // Precompute d-values for the first interval
  pchip_b_ptr = pchip_b;
  pchip_c_ptr = pchip_c;
  pchip_d_ptr = pchip_d;
  lookup_ptr_l = lookup;
  lookup_ptr_m = lookup + lookup_N;
  lookup_ptr_r = lookup + 2 * lookup_N;
  h1 = x_vals[1] - x_vals[0];
  h2 = x_vals[2] - x_vals[1];
  h1_inv = 1.0 / h1;
  h2_inv = 1.0 / h2;
  h1h2_inv = 1.0 / (h1 + h2);
  for (n = 0; n < lookup_N; n++, pchip_d_ptr++, lookup_ptr_l++, lookup_ptr_m++, lookup_ptr_r++)
  {
    pchip_delta1[n] = (*lookup_ptr_m - *lookup_ptr_l) * h1_inv; // / h1;
    pchip_delta2[n] = (*lookup_ptr_r - *lookup_ptr_m) * h2_inv; // / h2;
    *pchip_d_ptr = ((2.0 * h1 + h2) * pchip_delta1[n] - h1 * pchip_delta2[n]) * h1h2_inv; // / (h1 + h2);
    if (*pchip_d_ptr * pchip_delta1[n] <= 0)
      *pchip_d_ptr = 0.0;
    else if (pchip_delta1[n] * pchip_delta2[n] <= 0 && fabs(*pchip_d_ptr) > fabs(3.0 * pchip_delta1[n]))
      *pchip_d_ptr = 3.0 * pchip_delta1[n];
  }

  // Precompute d-values for intermediate intervals
  for (i = 1; i < lookup_L - 1; i++)
  {
    h1 = x_vals[i] - x_vals[i-1];
    h2 = x_vals[i+1] - x_vals[i];
    h1_inv = 1.0 / h1;
    h2_inv = 1.0 / h2;
    h1sq_inv = h1_inv * h1_inv;
    w1 = 2.0 * h2 + h1;
    w2 = h2 + 2.0 * h1;
    w1w2 = w1 + w2;

    // delta1 has already been computed in the previous iteration as delta2
    // delta2 has already been computed for first iteration (skip on first iteration)
    if (i > 1)
    {
      for (n = 0; n < lookup_N; n++, lookup_ptr_m++, lookup_ptr_r++)
      {
        pchip_delta2[n] = (*lookup_ptr_r - *lookup_ptr_m) * h2_inv; // / h2;
      }
    }

    for (n = 0; n < lookup_N; n++, pchip_b_ptr++, pchip_c_ptr++, pchip_d_ptr++)
    {
      if (pchip_delta1[n] * pchip_delta2[n] <= 0)
      {
        *pchip_d_ptr = 0.0;
      }
      else
      {
        *pchip_d_ptr = w1w2 / (w1 / pchip_delta1[n] + w2 / pchip_delta2[n]);
      }
      *pchip_b_ptr = (*(pchip_d_ptr - lookup_N) - 2.0 * pchip_delta1[n] + *pchip_d_ptr) * h1sq_inv; // / (h1 * h1);
      *pchip_c_ptr = (3.0 * pchip_delta1[n] - 2.0 * *(pchip_d_ptr - lookup_N) - *pchip_d_ptr) * h1_inv; // / h1;
    }

    // delta2 is delta1 in next iteration
    // delta2 in next iteration will be ovewritten in next iteration or reused in the last iteration
    pchip_delta_swap = pchip_delta1;
    pchip_delta1 = pchip_delta2;
    pchip_delta2 = pchip_delta_swap;
  }

  // Precompute d-values for the last interval
  h1 = x_vals[lookup_L - 1] - x_vals[lookup_L - 2]; // Note: delta1 and delta2 have already been swapped
  h2 = x_vals[lookup_L - 2] - x_vals[lookup_L - 3];
  h1_inv = 1.0 / h1;
  h1h2_inv = 1.0 / (h1 + h2);
  h1sq_inv = h1_inv * h1_inv;
  for (n = 0; n < lookup_N; n++, pchip_b_ptr++, pchip_c_ptr++, pchip_d_ptr++)
  {
    *pchip_d_ptr = ((2.0 * h1 + h2) * pchip_delta1[n] - h1 * pchip_delta2[n]) * h1h2_inv; // / (h1 + h2);
    if (*pchip_d_ptr * pchip_delta1[n] <= 0)
      *pchip_d_ptr = 0.0;
    else if (pchip_delta1[n] * pchip_delta2[n] <= 0 && fabs(*pchip_d_ptr) > fabs(3.0 * pchip_delta1[n]))
      *pchip_d_ptr = 3.0 * pchip_delta1[n];
    *pchip_b_ptr = (*(pchip_d_ptr - lookup_N) - 2.0 * pchip_delta1[n] + *pchip_d_ptr) * h1sq_inv; // / (h1 * h1);
    *pchip_c_ptr = (3.0 * pchip_delta1[n] - 2.0 * *(pchip_d_ptr - lookup_N) - *pchip_d_ptr) * h1_inv; // / h1;
  }
}

void initReluIdxLookup(int* approx_idx_relu_lookup, int approx_method,
    int lookup_L, double* x_vals)
{
  // Note: lookup_L is the number of x-values, i.e., approx_N + 1
  int i;
  if (approx_method == INTERPOLATE_NEAREST)
  {
    // Precompute approximation indices for nearest neighbor interpolation
    // +1 just to be on the safe side and we do not have to care about the boundary case
    int lookup_size = (int) (x_vals[lookup_L - 1] / VAL_2_POW_M4) + 1;
    int idx = 0;
    double cmp = VAL_2_POW_M4; // 0.5 * x_vals[1]

    approx_idx_relu_lookup[0] = 0;
    for (i = 0; i < lookup_size; i++)
    {
      double val = ((double) i + 0.5) * VAL_2_POW_M4; // intermediate value in interval of length 2^(-4)
      if (val > cmp)
      {
        idx++;
        if (idx == lookup_L - 1)
          cmp = 2.0 * x_vals[lookup_L - 1]; // everything to the right is rounded to the last value
        else
          cmp = 0.5 * (x_vals[idx] + x_vals[idx+1]); // compute intermediate value between next to x-values
      }
      approx_idx_relu_lookup[i] = idx;
    }
  }
  else if (approx_method == INTERPOLATE_LINEAR || approx_method == INTERPOLATE_PCHIP)
  {
    // Precompute approximation indices for linear and pchip interpolation
    // This is different to nearest neighbor interpolation since we now compute the
    // index of the x-value left to it rather than the index of the closest x-value.
    // All x to the right of x_vals[lookup_L - 2] have the (lookup_L-2)th point as their
    // left x-value.
    // +1 just to be on the safe side and we do not have to care about the boundary case
    int lookup_size = (int) (x_vals[lookup_L - 2] / VAL_2_POW_M3) + 1;
    approx_idx_relu_lookup[0] = 0;
    for (i = 1; i < lookup_size - 1; i++)
      approx_idx_relu_lookup[i] = log2int((unsigned int) i) + 1;
    approx_idx_relu_lookup[lookup_size - 1] = lookup_L - 2;
  }
  else
  {
    _printf("Error in samplez: Unknown interpolation (we should not be here)\n");
  }
}
